# %% [markdown]
# ### Perform bias correction on ERA5 daily TMAX & TMIN anomalies

# %%
# === load packages ===
from pathlib import Path
from scipy.stats import wasserstein_distance_nd
import h5py
import numpy as np
import netCDF4 as nc
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

# === parallel
import argparse
# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Process sub-region of the globe")
parser.add_argument("--region_id", type=int, required=True)
args = parser.parse_args()

region_id = args.region_id

# --- define spatial regions ---
# --- efficient for multi-yrs data processing
lat_bins = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
lon_bins = [(-180, -90), (-90, 0), (0, 90), (90, 180)]

regions = [(lat, lon) for lat in lat_bins for lon in lon_bins]

lat_range, lon_range = regions[region_id]

print(f"Processing region {region_id}: lat {lat_range} lon {lon_range}", flush=True)

# %%
# === config the data paths === 
DATA_DIR = Path("/ourdisk/hpc/ai2es/yxie/T_ghcnd_krig/")
ERA5_DIR = Path("/ourdisk/hpc/ai2es/yxie/ERA5/")
OUT_DIR = Path("/scratch/yxie/bias_corr_data/")

FIRST_YEAR = 2000
LAST_YEAR = 2025

FIRST_MONTH = 1
LAST_MONTH = 12
MONSTR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- choose bias correction method ---
#METHOD = "CDFt"
#METHOD = "ECBC"
#METHOD = "R2D2"
#METHOD = "MBCn"
METHOD = "TSMBC"

# %% [markdown]
# #### Self-defined helpers

# %%
def qc_temperature(arr, tmin=-89.0, tmax=56.0):
    """
    Clip unrealistic temperatures
    """
    ## keep the spatial shape (nlat, nlon)
    #nd_low = np.sum(arr < tmin, axis=0)
    #nd_high = np.sum(arr > tmax, axis=0)
    
    return np.clip(arr, tmin, tmax)


# === file list management ===
def monthly_file_list(start=f'{FIRST_YEAR:4d}{FIRST_MONTH:02d}', end=f'{LAST_YEAR:4d}{LAST_MONTH:02d}'):
    dates = pd.period_range(start, end, freq='M')
    return [str(DATA_DIR/f"tmnmx_pair_{p.strftime('%Y%m')}.h5") for p in dates]

# --- test out
files = monthly_file_list()
files[0]

# %%
# === load regional data from .h5 files === #
def load_region_from_h5(fname, lat_range, lon_range, qc=True):
    with h5py.File(fname, 'r') as f:
        lat = f['lat'][:]
        lon = f['lon'][:]

        lat_mask = (lat >= lat_range[0]) & (lat < lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon < lon_range[1])

        era_tmax = f['era_tmax'][:, lat_mask, :][:, :, lon_mask]
        era_tmin = f['era_tmin'][:, lat_mask, :][:, :, lon_mask]
        ref_tmax = f['ghcn_tmax'][:, lat_mask, :][:, :, lon_mask]
        ref_tmin = f['ghcn_tmin'][:, lat_mask, :][:, :, lon_mask]
        std_tmax = f['std_tmax'][:, lat_mask, :][:, :, lon_mask]
        std_tmin = f['std_tmin'][:, lat_mask, :][:, :, lon_mask]

        # apply post-processing quality control to clip unrealistic temperatures
        if qc:
            ref_tmax = qc_temperature(ref_tmax, tmin=-89.0, tmax=56.0)
            ref_tmin = qc_temperature(ref_tmin, tmin=-89.0, tmax=56.0)

        # construct time series
        year = int(fname[-9:-5])
        month = int(fname[-5:-3])
        ndays = era_tmax.shape[0]
        time = pd.date_range(f"{year}-{month:02d}-01", periods=ndays, freq='D')

    return lat[lat_mask], lon[lon_mask], time, era_tmax, era_tmin, ref_tmax, ref_tmin, std_tmax, std_tmin

# %%
# load the ERA5 land-sea mask
def load_landmask(fname):
    """Return (lat, lon, land_sea_mask) or None if not available"""
    if fname is None or not fname.exists():
        return None
    with nc.Dataset(fname, "r") as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        # land binary mask: 0-ocean; 1-land
        lsm = ds.variables["lsm"][0,:,:]
    return (np.array(lat), np.array(lon), np.array(lsm))

LSMASK = load_landmask(fname=ERA5_DIR/"ERA5reanalysis_landseamask.nc")
lat_era, lon_era0, lsm_era0 = LSMASK

# now convert land-sea mask from lon: 0-360 to lon: -180-180
idlon1 = np.where((lon_era0>=0)&(lon_era0<180))[0]
idlon2 = np.where((lon_era0>=180)&(lon_era0<360))[0]
lon_era = np.concatenate((lon_era0[idlon2]-360.0, lon_era0[idlon1]), axis=0)
# 0-ocean; 1-land
lsm_era = np.concatenate((lsm_era0[:, idlon2], lsm_era0[:, idlon1]), axis=1)
# determine the land-sea mask: True (1) for land / False (0) for sea
lsmask_era = (lsm_era >= 0.5)


def load_region_landmask(lsmask, lat_era, lon_era, lat_range, lon_range):
    lat_mask = (lat_era >= lat_range[0]) & (lat_era < lat_range[1])
    lon_mask = (lon_era >= lon_range[0]) & (lon_era < lon_range[1])
    lsmask_region = lsmask[lat_mask, :][:, lon_mask]

    return lat_era[lat_mask], lon_era[lon_mask], lsmask_region


# %%
def group_labels(time_index, group="month"):
    time_index = pd.DatetimeIndex(time_index)
    if group == "month":
        return time_index.month.values
    elif group == "season":
        m = time_index.month.values
        # DJF=1, MAM=2, JJA=3, SON=4
        return ((m%12) // 3) + 1
    else:
        raise ValueError("group must be 'month' or 'season'")

# %%
# === monthly climatology & anomalies ===
def compute_monthly_anomaly(data, time_index):
    nt = data.shape[0]
    flat = data.reshape(nt, -1)
    df = pd.DataFrame(flat, index=time_index)
    
    # monthly climatology
    clim = df.groupby(df.index.month).mean()
    # map monthly climatology to daily
    month_idx = df.index.month.values - 1
    clim_daily = clim.values[month_idx, :]
    # compute anomaly
    anom = flat - clim_daily
    
    return anom.reshape(data.shape), clim_daily, clim

# %%
def random_orthogonal_matrix(ndim, rng):
    """
    Generate a random orthogonal matrix using QR decomposition
    """
    A = rng.normal(size=(ndim, ndim))
    Q, R = np.linalg.qr(A)
    return Q

# %%
def hist_nd(X, nbins=50):
    """Estimate histogram along sample-axis"""
    ntime, ndim = X.shape
    X_hist = np.full((nbins, ndim), np.nan)
    for idim in range(0, ndim):
        # only estimate histogram on finite values
        X_temp = X[:, idim].copy()
        X_temp = X_temp[np.isfinite(X_temp)]
        X_hist[:, idim], _ = np.histogram(X_temp, bins=nbins, density=True)
 
    return X_hist

# %%
def quantile_map_1d(x, y):
    """
    Map x -> y using empirical quantile mapping
    NaNs in y are ignore; NaNs in x are kept

    Parameters
    ----------
    x : (n, ) np.ndarray
    y : (n, ) np.ndarray (may contain NaNs)

    Returns
    ----------
    x_mapped : (n, ) np.ndarray 
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_out = x.copy()
    mask_x = ~np.isnan(x)
    mask_y = ~np.isnan(y)

    if mask_x.sum()==0 or mask_y.sum()==0:
        return x_out
    
    x_valid = x[mask_x]
    y_valid = y[mask_y]

    # empirical CDF of x
    ranks = stats.rankdata(x_valid, method='average')
    q = (ranks-0.5) / len(x_valid)

    # empirical quantiles of y
    y_sorted = np.sort(y_valid)
    qy = (np.arange(1, len(y_sorted)+1)-0.5) / len(y_sorted)

    # quantile mapping
    x_out[mask_x] = np.interp(q, qy, y_sorted)

    return x_out


# %%
def qm_each_margin(X, Y):
    """
    Univariate empirical quantile mapping applied column by column
    X, Y: shape (nt, d)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Xc = X.copy()
    # correction for each variable separately
    for j in range(X.shape[1]):
        Xc[:, j] = quantile_map_1d(X[:, j], Y[:, j])    
    
    return Xc

# %%
# === sufficient GHCN-d observations to initiate bias correction === #
def sufficient_obs(ts, min_days=1600):
    return np.isfinite(ts).sum() >= min_days

# %%
def wasserstein_metric(X, Y):
    """
    X, Y: arrays of shape (ntime, ndim)
    """
    # remove NaNs pairwise
    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(Y).any(axis=1))
    if len(mask) == 0:
        return np.nan
    
    Xv = X[mask]
    Yv = Y[mask]
    wdist = wasserstein_distance_nd(Xv, Yv)

    return wdist

# %%
def make_lagged_matrix(X, lags=(0, 1, 3, 7)):
    """
    Build lag-expanded matrix
    example: [x_t, y_t, x_t-i, y_t-i, ...] i=1, 3, 7, etc.

    Returns
    -------
    MX : (nt-maxlag, d*len(lags))
    maxlag: int
    """
    X = np.asarray(X, float)
    nt, d = X.shape
    maxlag = max(lags)
    
    mats = []
    for lag in lags:
        mats.append(X[maxlag - lag : nt -lag, :])

    MX = np.concatenate(mats, axis=1)
    return MX, maxlag

# %%
class SlopeStopping:
    def __init__(self, min_iter=10, max_iter=50, tol=1e-4, window=5):
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        # sliding window to estimate local trend for convergence estimation
        self.window = window
        self.history = []

    def update(self, value):
        self.history.append(value)
    
    def should_stop(self):
        n = len(self.history)
        if n < self.min_iter:
            return False
        if n >= self.max_iter:
            return True
        if n < self.window + 1:
            return False
        
        y = np.array(self.history[-self.window:])
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        flag_stop = np.abs(slope) < self.tol
        
        return flag_stop

# %%
class PatienceStopping:
    def __init__(self, min_iter=20, max_iter=100, rel_tol=1e-5, patience=10, ema_alpha=0.3):
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.patience = patience
        self.ema_alpha = ema_alpha
        
        self.history = []
        self.best = np.inf
        self.best_iter = -1
        self.ema = None

    def update(self, value):
        if self.ema is None:
            self.ema = float(value)
        else:
            # optional EMA smoothing
            self.ema = self.ema_alpha * float(value) + (1 - self.ema_alpha) * self.ema
        
        v = self.ema
        self.history.append(v)

        if v < self.best * (1 - self.rel_tol):
            self.best = v
            self.best_iter = len(self.history) - 1
    
    def should_stop(self):
        n = len(self.history)
        
        if n < self.min_iter:
            return False
        if n >= self.max_iter:
            return True
        
        # or stop if no meaningful improvements after patience iterations
        if (n-1) - self.best_iter >= self.patience:
            return True
        
        return False
        

# %% [markdown]
# #### CDF-t helper
# CDF-t maps the model distribution to the reference distribution through a transfer in cumulative distribution function space. The pratical implementation is just univariate empirical quantile mapping on each variable separately.

# %%
def cdft_single_period_stationary(
        X, Y,
        time_index,
        group="season",
        min_group_n=400):
    """
    Stationary/historical-period CDF-t is operationally equivalent
    to empirical quantile mapping applied separately to TMAX & TMIN

    Parameters
    ----------
    X: (nt, 2) ERA5 anomalies
    Y: (nt, 2) GHCN-d anomalies

    Returns
    ----------
    Xc: (nt, 2) corrected anomalies
    """
    Xc = X.copy()

    labels = group_labels(time_index, group)

    for g in np.unique(labels):
        idx = np.where(labels==g)[0]

        valid = idx[
            np.all(np.isfinite(X[idx, :]), axis=1) &
            np.all(np.isfinite(Y[idx, :]), axis=1)
        ]
        # only apply bias correction when there is sufficient data samples for the specific group
        if valid.size < min_group_n:
            continue

        Xc[valid, :] = qm_each_margin(X[valid, :], Y[valid, :])

    return Xc

# %% [markdown]
# #### EC-BC helper
# EC-BC main steps include: (1) univariate bias correction for each margin, and (2) empirical coupla coupling via reordering the corrected marginals to restore the reference multivariate dependence

# %%
def ecbc_single_period(
        X, Y,
        time_index,
        group="season",
        min_group_n=400,
    ):
    """
    EC-BC for bivariate anomalies:
    1) univariate QM on each margin
    2) empirical-copula coupling by rank reordering within each group (month/season) separately for TMAX & TMIN

    Parameters
    ----------
    X: (nt, 2) ERA5 anomalies
    Y: (nt, 2) GHCNd anomalies
    time_index: DatetimeIndex
    group: "month" or "season"
    min_group_n: minimum valid sample size per group

    Returns
    ----------
    out: (nt, 2) corrected anomalies
    """
    Xuni = X.copy()
    out = X.copy()

    labels = group_labels(time_index, group)

    for g in np.unique(labels):
        idx = np.where(labels==g)[0]
        valid = idx[
            np.all(np.isfinite(X[idx, :]), axis=1) &
            np.all(np.isfinite(Y[idx, :]), axis=1)
        ]
        # only apply bias correction when there is sufficient data samples for the specific group
        if valid.size < min_group_n:
            continue
        
        # (1) correct univariate marginal distribution
        Xuni[valid, :] = qm_each_margin(X[valid, :], Y[valid, :])
        
        # (2) correct empirical-copula coupling
        for j in range(Xuni.shape[1]):
            ref_rank = stats.rankdata(Y[valid, j], method="ordinal") - 1
            sorted_vals = np.sort(Xuni[valid, j])
            out[valid, j] = sorted_vals[ref_rank] 
    
    return out

# %% [markdown]
# #### R2-D2 helper

def r2d2_single_period_ensemble(
        X, Y,
        time_index,
        group="season",
        min_group_n=400,
        random_state=42,
        n_ens = 50,
):
    """
    Original R2D2 algorithm (Vrac & Thao 2020)

    Parameters
    -----------
    X: (nt, 2) ERA5 anomalies
    Y: (nt, 2) GHCNd anomalies
    rng: numpy random generator
    n_ens: number of ensemble realizations

    Returns
    -----------
    Xc: (n_ens, nt, 2) bias-corrected 
    """
    rng = np.random.default_rng(random_state)
    
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    nt_all, d = X.shape
    out = np.full((n_ens, nt_all, d), np.nan, dtype=float)
    
    labels = group_labels(time_index, group)

    for g in np.unique(labels):
        idx = np.where(labels==g)[0]
        valid = idx[
            np.all(np.isfinite(X[idx, :]), axis=1) &
            np.all(np.isfinite(Y[idx, :]), axis=1)
        ]
        
        # only apply bias correction when there is sufficient data samples for the specific group
        if valid.size < min_group_n:
            continue
        
        Xg = X[valid, :]
        Yg = Y[valid, :]
        nt = Xg.shape[0]

        # (1) correct univariate marginal distribution
        Xuni_base = qm_each_margin(Xg, Yg)
        
        # (2) reference ranks
        ref_ranks = np.empty((nt, d), dtype=int)
        for j in range(d):
            ref_ranks[:, j] = stats.rankdata(Yg[:, j], method="ordinal") - 1
        
        # (3) sort corrected marginals
        sorted_vals = np.sort(Xuni_base, axis=0) # anchor to daily TMAX
        
        # (4) ensemble resampling of rank vectors
        for e in range(n_ens):
            Xe = np.empty_like(Xuni_base)
            sampled_idx = rng.integers(0, nt, size=nt)
            for i in range(nt):
                k = sampled_idx[i]
                for j in range(d):
                    r = ref_ranks[k, j]
                    Xe[i, j] = sorted_vals[r, j]
            
            out[e, valid, :] = Xe
    
    return out

# %% [markdown]
# #### MBCn helper

# %%
def mbcn_single_period_wasserstein(
        X, Y,
        time_index,
        group="season",
        min_group_n=400,
        random_state = 42,
        stopping_params=None,
        verbose=False,
    ):
    """
    MBCn with Wasserstein-based stopping
    Returns
    ----------
    X_corr: corrected data
    history: Wasserstein distance history
    """
    if stopping_params is None:
        stopping_params = dict(min_iter=15, max_iter=80, rel_tol=1e-5, patience=10)
    
    # target dataset
    X = np.asarray(X, float)
    # reference dataset
    Y = np.asarray(Y, float)

    rng = np.random.default_rng(random_state)
    ntime, ndim = X.shape
    
    Xc = X.copy()
    
    labels = group_labels(time_index, group)

    for g in np.unique(labels):

        idx = np.where(labels==g)[0]
        valid = idx[
            np.all(np.isfinite(X[idx, :]), axis=1) &
            np.all(np.isfinite(Y[idx, :]), axis=1)
        ]
        # only apply bias correction when there is sufficient data samples for the specific group
        if valid.size < min_group_n:
            continue
        
        # renew the history records
        stopper = PatienceStopping(**stopping_params)

        for iter in range(stopper.max_iter):
            # step1 : rotate and project into orthogonal matrix
            R = random_orthogonal_matrix(ndim, rng)
            Xr = Xc[valid, :] @ R
            Yr = Y[valid, :] @ R
        
            # step 2: bias correction for each dimension/variable/feature
            for j in range(ndim):
                Xr[:, j] = quantile_map_1d(Xr[:, j], Yr[:, j])
        
            # step 3: rotate back
            # for orthogonal matrix, inverse equals to transpose
            Xc[valid, :] = Xr @ (R.T)

            # step 4: calculate wasserstein distance approximate
            # --- first estimate histogram --- #
            Xc_hist = hist_nd(Xc[valid, :], nbins=40)
            Y_hist = hist_nd(Y[valid, :], nbins=40)
            
            # calculate the wasserstein distance
            d = wasserstein_metric(Xc_hist, Y_hist)
            stopper.update(d)

            # step 5: progress & convergence
            if verbose:
                print(f"MBCn Iter {iter+1:02d} | W_approx = {d:.6f}", flush=True)

            if stopper.should_stop():
                print(f"MBCn Iter {iter+1:02d}: Stopping criterion met", flush=True)
                break

    return Xc


# %% [markdown]
# #### TSMBC helper
# TSMBC treats lagged copies of the variables as additional variables, then applies a base multivariate BC method in that extended space

# %%
def tsmbc_single_period_mbcn(
        X, Y, 
        time_index,
        group="season",
        min_group_n=300,
        lags = (0, 1),
        stopping_params=None,
        random_state=42,
):
    """
    TSMBC using the existing MBCn / ECBC

    Parameters
    ----------
    X: (nt, 2) ERA5 anomalies
    Y: (nt, 2) GHCNd anomalies
    lags: tuple of int
    stopping_params: passed to mbcn_single_period_wasserstein
    random_state: int

    Returns
    ----------
    out: (nt, 2) corrected anomalies
    """
    if stopping_params is None:
        stopping_params = dict(min_iter=15, max_iter=80, rel_tol=1e-5, patience=10)

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    nt, d = X.shape

    MX, maxlag = make_lagged_matrix(X, lags=lags)
    MY, _ = make_lagged_matrix(Y, lags=lags)
    
    timelag = time_index[maxlag : nt]
   
    # only use where both lag-expanded samples are finite
    valid = np.all(np.isfinite(MX), axis=1) & np.all(np.isfinite(MY), axis=1)

    out = np.full_like(X, np.nan)
    history = None

    if valid.sum() == 0:
        return out
    
    #MZ = mbcn_single_period_wasserstein(
    #    X=MX[valid],
    #    Y=MY[valid],
    #    time_index=timelag[valid],
    #    group=group,
    #    min_group_n=min_group_n,
    #    random_state=random_state,
    #    stopping_params=stopping_params,
    #    verbose=False,
    #)

    MZ = ecbc_single_period(
        X=MX[valid], 
        Y=MY[valid], 
        time_index=timelag[valid],
        group=group, 
        min_group_n=min_group_n
    )

    
    # lag-0 columns are the first d columns
    cor_rows = np.where(valid)[0] + maxlag
    out[cor_rows, :] = MZ[:, :d]

    ## fill the first few days with simple marginal QM
    #if maxlag > 0:
    #    out[:maxlag, :] = qm_each_margin(X[:maxlag, :], Y[:maxlag, :])

    return out   

# %% [markdown]
# #### Grid-wise bias correction

# %%
if __name__ == "__main__":

    lat_range, lon_range = regions[region_id]
    files = monthly_file_list()
    
    # --- pre-allocate
    ERA_TMAX, ERA_TMIN, REF_TMAX, REF_TMIN = [], [], [], []
    TIME = []

    # --- loop through months
    for f in files:
        try:
            # --- load the data
            lat_reg, lon_reg, time, era_tmax, era_tmin, ref_tmax, ref_tmin, std_tmax, std_tmin = load_region_from_h5(f, lat_range, lon_range, qc=True)
            lat_reg0, lon_reg0, lsmask_region = load_region_landmask(lsmask_era, lat_era, lon_era, lat_range, lon_range)
            
            # --- ensure spatial range consistency
            assert (lat_reg[0] == lat_reg0[0]) and (lat_reg.shape == lat_reg0.shape) and (lat_reg[-1] == lat_reg0[-1])
            assert (lon_reg[0] == lon_reg0[0]) and (lon_reg.shape == lon_reg0.shape) and (lon_reg[-1] == lon_reg0[-1])

            # --- apply land-sea mask to ERA5 data
            era_tmax = era_tmax * lsmask_region
            era_tmin = era_tmin * lsmask_region
            
            # --- concatenate along the time axis
            ERA_TMAX.append(era_tmax)
            ERA_TMIN.append(era_tmin)
            REF_TMAX.append(ref_tmax)
            REF_TMIN.append(ref_tmin)
            TIME.append(time)
        
        except Exception as e:
            print(f"[Fail load data] for {f}: {e}")

    
    print("Imports done", flush=True)

    ERA_TMAX = np.concatenate(ERA_TMAX, axis=0)
    ERA_TMIN = np.concatenate(ERA_TMIN, axis=0)
    REF_TMAX = np.concatenate(REF_TMAX, axis=0)
    REF_TMIN = np.concatenate(REF_TMIN, axis=0)
    TIME = pd.DatetimeIndex(np.concatenate(TIME)) 

    assert len(TIME) == ERA_TMAX.shape[0]

    # --- calculate daily temperature range (DTR) --- #
    # --- DTR must >= 0
    ERA_DTR = ERA_TMAX - ERA_TMIN
    REF_DTR = REF_TMAX - REF_TMIN
    # --- clean out invalid data points
    MASK_ERA = np.isfinite(ERA_DTR) & (ERA_DTR < 0)
    ERA_DTR[MASK_ERA] = np.nan
    
    MASK_REF = np.isfinite(REF_DTR) & (REF_DTR < 0)
    REF_DTR[MASK_REF] = np.nan

    # add a small additional term to account for DTR=0 situation
    epsilon = 0.001
    ERA_DTR += epsilon
    REF_DTR += epsilon
     
    # --- compute anomalies 
    era_tmax_anom, era_tmax_climd, _ = compute_monthly_anomaly(ERA_TMAX, TIME)   
    era_logdtre_anom, era_logdtre_climd, _ = compute_monthly_anomaly(np.log(ERA_DTR), TIME)
    ref_tmax_anom, ref_tmax_climd, _ = compute_monthly_anomaly(REF_TMAX, TIME)
    ref_logdtre_anom, ref_logdtre_climd, _ = compute_monthly_anomaly(np.log(REF_DTR), TIME)

    ## rescale to compress
    #scale = 10.0
    #ERA_DTR_soft = np.log(1 + np.exp(ERA_DTR / scale))
    #REF_DTR_soft = np.log(1 + np.exp(REF_DTR / scale))
    # 
    ## --- compute anomalies 
    #era_tmax_anom, era_tmax_climd, _ = compute_monthly_anomaly(ERA_TMAX, TIME)   
    #era_softdtre_anom, era_softdtre_climd, _ = compute_monthly_anomaly(ERA_DTR_soft, TIME)
    #ref_tmax_anom, ref_tmax_climd, _ = compute_monthly_anomaly(REF_TMAX, TIME)
    #ref_softdtre_anom, ref_softdtre_climd, _ = compute_monthly_anomaly(REF_DTR_soft, TIME)

    nt, nlat, nlon = ERA_TMAX.shape

    if METHOD == "R2D2":
        N_ENS = 20
        tmax_corr_ens = np.full((N_ENS, nt, nlat, nlon), np.nan, dtype=np.float32)
        tmin_corr_ens = np.full((N_ENS, nt, nlat, nlon), np.nan, dtype=np.float32)
    
    elif METHOD != "R2D2":
        tmax_corr = np.full((nt, nlat, nlon), np.nan)
        tmin_corr = np.full((nt, nlat, nlon), np.nan)


    # apply bias correction per grid
    for i in tqdm(range(len(lat_reg))):
        for j in range(len(lon_reg)):
            # --- bias correction only over land grids ---
            if lsmask_region[i, j] < 0.5:
                continue
            
            # --- time series for the grid --- 
            r1 = ref_tmax_anom[:, i, j]
            r2 = ref_logdtre_anom[:, i, j]
            s1 = era_tmax_anom[:, i, j]
            s2 = era_logdtre_anom[:, i, j]

            ## --- time series for the grid --- 
            #r1 = ref_tmax_anom[:, i, j]
            #r2 = ref_softdtre_anom[:, i, j]
            #s1 = era_tmax_anom[:, i, j]
            #s2 = era_softdtre_anom[:, i, j]

            # --- only perform MBCn with sufficient GHCNd observations ---
            if not (sufficient_obs(r1) and sufficient_obs(r2)):
                continue

            X = np.vstack([s1, s2]).T
            Y = np.vstack([r1, r2]).T

            # --- bias correction on anomalies distribution
            if METHOD == "CDFt":
                era_anom_corr = cdft_single_period_stationary(
                    X=X, Y=Y, time_index=TIME,
                    group="season", min_group_n=300)
            
            elif METHOD == "ECBC":
                era_anom_corr = ecbc_single_period(
                    X=X, Y=Y, time_index=TIME,
                    group="season", min_group_n=300)

            elif METHOD == "R2D2":
                era_anom_corr_ensem = r2d2_single_period_ensemble(
                    X=X, Y=Y, time_index=TIME,
                    group="season", min_group_n=300, random_state=42, n_ens=N_ENS)

            elif METHOD == "MBCn":
                era_anom_corr = mbcn_single_period_wasserstein(
                    X=X, Y=Y,time_index=TIME,
                    group="season", min_group_n=300, random_state = 42,
                    stopping_params=dict(min_iter=15, max_iter=100, rel_tol=1e-5, patience=10), verbose=False)

            elif METHOD == "TSMBC":
                era_anom_corr = tsmbc_single_period_mbcn(
                    X=X, Y=Y, time_index=TIME,
                    group="season", min_group_n=300, lags = (0, 1), 
                    stopping_params=dict(min_iter=15, max_iter=100, rel_tol=1e-5, patience=10), random_state=42)
                
            else:
                raise ValueError(f"Unkown METHOD: {METHOD}")
            
            if METHOD != "R2D2":
                # --- add back the climatology ---
                # retrieve the flattened index
                tmax_clim_daily = ref_tmax_climd[:, i*len(lon_reg)+j]
                logdtre_clim_daily = ref_logdtre_climd[:, i*len(lon_reg)+j]
                #softdtre_clim_daily = ref_softdtre_climd[:, i*len(lon_reg)+j]

                # add back the climatology
                tmax_corr[:, i, j] = era_anom_corr[:, 0] + tmax_clim_daily
                dtr = np.exp( era_anom_corr[:, 1] + logdtre_clim_daily ) - epsilon
                #dtr = scale * np.log( np.exp(era_anom_corr[:, 1] + softdtre_clim_daily) - 1 )
                tmin_corr[:, i, j] = tmax_corr[:, i, j] - dtr

                
            elif METHOD == "R2D2": 
                # retrieve the flattened index
                tmax_clim_daily = ref_tmax_climd[:, i*len(lon_reg)+j]
                logdtre_clim_daily = ref_logdtre_climd[:, i*len(lon_reg)+j]
                #softdtre_clim_daily = ref_softdtre_climd[:, i*len(lon_reg)+j]

                for e in range(N_ENS):
                    tmax_corr_e = era_anom_corr_ensem[e, :, 0] + tmax_clim_daily
                    dtr_e = np.exp( era_anom_corr_ensem[e, :, 1] + logdtre_clim_daily ) - epsilon
                    #dtr_e = scale * np.log( np.exp(era_anom_corr_ensem[e, :, 1] + softdtre_clim_daily) - 1 )
                    tmin_corr_e = tmax_corr_e - dtr_e

                    tmax_corr_ens[e, :, i, j] = tmax_corr_e
                    tmin_corr_ens[e, :, i, j] = tmin_corr_e

    
    # save into output file
    fout = OUT_DIR/f"TmxmnCorr_{METHOD}_region{region_id:02d}.h5"
    
    if METHOD != "R2D2":
        with h5py.File(fout, "w") as hout:
            hout.create_dataset("lat_reg", data=lat_reg, compression="lzf", dtype="float32")
            hout.create_dataset("lon_reg", data=lon_reg, compression="lzf", dtype="float32")
            hout.create_dataset("time", data=TIME.view("int64"), compression="lzf")
            # when read back: TIME = pd.to_datetime(time_ns)
            # ERA raw data
            hout.create_dataset("era_tmax", data=ERA_TMAX, compression="lzf", dtype="float32")
            hout.create_dataset("era_tmin", data=ERA_TMIN, compression="lzf", dtype="float32")
            hout.create_dataset("era_tmax_climd", data=era_tmax_climd, compression="lzf", dtype="float32")
            hout.create_dataset("era_logdtre_climd", data=era_logdtre_climd, compression="lzf", dtype="float32")
            #hout.create_dataset("era_softdtre_climd", data=era_softdtre_climd, compression="lzf", dtype="float32")
            # GHCN reference data
            hout.create_dataset("ref_tmax", data=REF_TMAX, compression="lzf", dtype="float32")
            hout.create_dataset("ref_tmin", data=REF_TMIN, compression="lzf", dtype="float32")
            hout.create_dataset("ref_tmax_climd", data=ref_tmax_climd, compression="lzf", dtype="float32")
            hout.create_dataset("ref_logdtre_climd", data=ref_logdtre_climd, compression="lzf", dtype="float32")
            #hout.create_dataset("ref_softdtre_climd", data=ref_softdtre_climd, compression="lzf", dtype="float32")
            # corrected ERA data
            hout.create_dataset("corr_tmax", data=tmax_corr, compression="lzf", dtype="float32")
            hout.create_dataset("corr_tmin", data=tmin_corr, compression="lzf", dtype="float32")

    elif METHOD == "R2D2":
        with h5py.File(fout, "w") as hout:
            hout.create_dataset("lat_reg", data=lat_reg, compression="lzf", dtype="float32")
            hout.create_dataset("lon_reg", data=lon_reg, compression="lzf", dtype="float32")
            hout.create_dataset("time", data=TIME.view("int64"), compression="lzf")
            # when read back: TIME = pd.to_datetime(time_ns)
            # ERA raw data
            hout.create_dataset("era_tmax", data=ERA_TMAX, compression="lzf", dtype="float32")
            hout.create_dataset("era_tmin", data=ERA_TMIN, compression="lzf", dtype="float32")
            hout.create_dataset("era_tmax_climd", data=era_tmax_climd, compression="lzf", dtype="float32")
            hout.create_dataset("era_logdtre_climd", data=era_logdtre_climd, compression="lzf", dtype="float32")
            #hout.create_dataset("era_softdtre_climd", data=era_softdtre_climd, compression="lzf", dtype="float32")
            # GHCN reference data
            hout.create_dataset("ref_tmax", data=REF_TMAX, compression="lzf", dtype="float32")
            hout.create_dataset("ref_tmin", data=REF_TMIN, compression="lzf", dtype="float32")
            hout.create_dataset("ref_tmax_climd", data=ref_tmax_climd, compression="lzf", dtype="float32")
            hout.create_dataset("ref_logdtre_climd", data=ref_logdtre_climd, compression="lzf", dtype="float32")
            #hout.create_dataset("ref_softdtre_climd", data=ref_softdtre_climd, compression="lzf", dtype="float32")
            # corrected ERA data
            hout.create_dataset("corr_tmax", data=np.nanmean(tmax_corr_ens,axis=0), compression="lzf", dtype="float32")
            hout.create_dataset("corr_tmin", data=np.nanmean(tmin_corr_ens,axis=0), compression="lzf", dtype="float32")
            hout.create_dataset("std_tmax", data=np.nanstd(tmax_corr_ens, axis=0), compression="lzf", dtype="float32")
            hout.create_dataset("std_tmin", data=np.nanstd(tmin_corr_ens, axis=0), compression="lzf", dtype="float32")


