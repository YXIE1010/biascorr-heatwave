# %%
# load modules
import numpy as np
import pandas as pd
import netCDF4 as nc
import h5py
from pathlib import Path
import calendar
import xarray as xr
import pickle
import argparse

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="lead day")
parser.add_argument("--days", type=int, required=True)
args = parser.parse_args()
lead_day = args.days

# --- define reference period: 2020-01-01 to 2025-12-31
years = range(2020, 2026)
months = range(1, 13)
lat_range, lon_range = ((-90, 90.25), (-180, 180))

TRUE_DIR = Path("/ourdisk/hpc/ai2es/yxie/T_ghcnd_krig")
FORECAST_OG_DIR = Path("/ourdisk/hpc/ai2es/yxie/credit_last_experiment/ERA5_og/forecasts")
FORECAST_CORR_DIR = Path("/ourdisk/hpc/ai2es/yxie/credit_last_experiment/ERA5_corr/forecasts")
P90_DIR = Path("/ourdisk/hpc/ai2es/yxie/")
ERA5_DIR = Path("/ourdisk/hpc/ai2es/yxie/ERA5/")
OUT_DIR = Path("/ourdisk/hpc/ai2es/yxie/")

NY, NX = 721, 1440
MIN_DAYS = 3
TYPE_HEAT = "comp"  # heatwave types: "day", "night", "comp"

# %% [markdown]
# #### Self-defined helpers

# %%
def qc_temperature(arr, tmin=-89.0, tmax=56.0):
    """
    Clip unrealistic temperatures
    """
    return np.clip(arr, tmin, tmax)

# %%
def load_truth_for_date(date, h5_dir, qc=True):
    y, m, d = date.year, date.month, date.day
    fname = h5_dir / f"tmnmx_pair_{y:4d}{m:02d}.h5"

    # sanity check
    dim = calendar.monthrange(y, m)[1] # number of days in a month
    k = d - 1
    if not( 0<= k <dim):
        return None, None, None, None

    with h5py.File(fname, "r") as h:
        # double sanity check
        if k >= h["ghcn_tmax"].shape[0]:
            return None, None, None, None

        tmax_obs = h["ghcn_tmax"][k, :, :].astype(np.float32)
        tmin_obs = h["ghcn_tmin"][k, :, :].astype(np.float32)

        # apply post-processing quality control to clip unrealistic temperatures
        if qc:
            tmax_obs = qc_temperature(tmax_obs)
            tmin_obs = qc_temperature(tmin_obs)

    return tmax_obs, tmin_obs

# %%
def doy_index_366(times):
    """
    map pandas datetime index onto 0...365
    matching 366-day climatology
    """
    idx = np.empty(len(times), dtype=np.int32)
    
    for i, t in enumerate(times):
        doy = t.dayofyear - 1
        leap = t.is_leap_year

        if (not leap) and (t.month>=3):
            doy+=1
        
        idx[i] = doy
    
    return idx

# %%
def summer_mask(time, lat):
    """
    Returns mask of shape (T, nlat, 1)
    """
    T = len(time)
    nlat = len(lat)
    months = pd.DatetimeIndex(time).month.values
    
    # NH: May - Sep
    nh = (months >=5) & (months<=9)
    # SH: Nov - Mar
    sh = (months >= 11) | (months<=3)

    mask = np.zeros((T, nlat), dtype=bool)
    for i, la in enumerate(lat):
        if la >= 0:
            mask[:, i] = nh
        else:
            mask[:, i] = sh
    
    return mask[:, :, None]  # broadcast over longitude


# %%
def consecutive_filter(exceed, min_days=MIN_DAYS):
    """
    retain only cells
    belonging to >=min_days runs
    """

    T=exceed.shape[0]
    run=np.ones_like(exceed, dtype=bool)

    for k in range(min_days):
        shifted=np.zeros_like(exceed, dtype=bool)
        shifted[:T-k]=exceed[k:]
        run &= shifted

    mask=np.zeros_like(exceed, dtype=bool)

    for k in range(min_days):
        mask[k:] |= run[:T-k]

    return mask

# %%
def detect_days(time, lat, tmax, tmin, p90_tmax, p90_tmin, min_len=3, type=TYPE_HEAT):
    """
    returns mask where points belong to 3+ consecutive days
    """

    if type == "day":
        exceed = (tmax > p90_tmax)
        anom = (tmax - p90_tmax)
    elif type == "night":
        exceed = (tmin > p90_tmin)
        anom = (tmin - p90_tmin)
    elif type == "comp":
        exceed = (tmax > p90_tmax) & (tmin > p90_tmin)
        anom = (tmax - p90_tmax) + (tmin - p90_tmin)
    
    print(f"exceedance mask percent: {np.sum(exceed) / np.sum(np.isfinite(tmax))}", flush=True)

    # apply the seasonal restriction
    # NH: May - Sep only ; SH: Nov - Mar only
    season_mask = summer_mask(time, lat)
    exceed = exceed & season_mask

    print(f"summer exceedance mask percent: {np.sum(exceed) / np.sum(np.isfinite(tmax))}", flush=True)

    mask = consecutive_filter(exceed, min_len)
    print(f"filtered mask percent: {np.sum(mask) / np.sum(np.isfinite(tmax))}", flush=True)

    return mask, anom


# %%
def metrics_from_mask(lat, lon, mask, anom):
    """
    calculate metrics from the filtered heat wave mask
    metrics include heat wave days, cumulative excess heat stress,
    number of heat wave events, maximum/longest heat wave duration
    """
    hw_days = np.sum(mask, axis=0).astype(np.int16)
    cum_heat = np.sum(np.where(mask, anom, 0), axis=0).astype(np.float32)

    nlat = len(lat)
    nlon = len(lon)
    n_events = np.zeros((nlat, nlon), dtype=np.int16)
    max_dura = np.zeros((nlat, nlon), dtype=np.int16)

    for ilat in range(0, nlat):
        for ilon in range(0, nlon):
            grid_mask = mask[:, ilat, ilon].astype(int)

            if np.sum(grid_mask) == 0:
                continue

            gap_ini = np.where(np.diff(grid_mask) == 1)[0] + 1
            gap_end = np.where(np.diff(grid_mask) == -1)[0]
            
            # deal with edge cases
            if grid_mask[0]:
                gap_ini = np.r_[0, gap_ini]
            
            if grid_mask[-1]:
                gap_end = np.r_[gap_end, len(grid_mask)-1]

            if len(gap_ini)==0 or len(gap_end)==0:
                continue

            durations = gap_end - gap_ini + 1

            n_events[ilat, ilon] = len(durations)
            max_dura[ilat, ilon] = np.max(durations)
    
    #print(f"Heat wave days range (min, max): ({np.min(hw_days)}, {np.max(hw_days)})")
    #print(f"Cumulative excess heat range (min, max): ({np.min(cum_heat)}, {np.max(cum_heat)})")
    #print(f"Heat wave event count range (min, max): ({np.min(n_events)}, {np.max(n_events)})")
    #print(f"Heat wave maximum duration days range (min, max): ({np.min(max_dura)}, {np.max(max_dura)})")

    metrics = {
        "hw_days": hw_days,
        "cum_heat": cum_heat,
        "n_events": n_events,
        "max_dura": max_dura
    }

    return metrics

# %%
#  Detect heat wave days on grid-wise level

# --- load observational climatology quantiles
fin = str(P90_DIR / "p90_obst_clim.h5")
with h5py.File(fin, "r") as f:
    lat_glob = f["lat"][:]
    lon_glob = f["lon"][:]
    p90_tmax_doy = f["ptl_tmax_doy"][:]    # shape (366, nlat, nlon)
    p90_tmin_doy = f["ptl_tmin_doy"][:]


metrics_obs_all = {}
metrics_og_all = {}
metrics_corr_all = {}
TP_og_all, FP_og_all, TN_og_all, FN_og_all = {}, {}, {}, {}
TP_corr_all, FP_corr_all, TN_corr_all, FN_corr_all = {}, {}, {}, {}

for iyear in years:
    
    # initialization dates for the forecasts
    init_dates = pd.date_range(f"{iyear:4d}-01-01", f"{iyear:4d}-12-31", freq="D")
    N_init = len(init_dates)
    
    # for a given lead day, determine the corresponding true dates
    obs_dates = init_dates + pd.Timedelta(days=lead_day)
    obs_dates_exist = obs_dates.copy()
    obs_dates_exist = obs_dates_exist[obs_dates_exist <= pd.Timestamp("2025-12-31")]

    # --- Step 1: get the true values 
    obs_tmax = np.full((N_init, NY, NX), np.nan, dtype=np.float32)
    obs_tmin = np.full((N_init, NY, NX), np.nan, dtype=np.float32)
    # loop over all initialization dates
    for i, init in enumerate(obs_dates_exist):   
        # load truth for the initialization date
        tmax_true, tmin_true = load_truth_for_date(init, TRUE_DIR)
        if tmax_true is not None:
            obs_tmax[i, :, :] = tmax_true
            obs_tmin[i, :, :] = tmin_true
    
    obs_time = pd.DatetimeIndex(obs_dates)

    # --- detect using doy-time climatology
    idx_doy = doy_index_366(obs_time)
    p90tmax_thres = p90_tmax_doy[idx_doy]
    p90tmin_thres = p90_tmin_doy[idx_doy]

    hw_mask_obs, anom_obs = detect_days(time=obs_time, lat = lat_glob, tmax=obs_tmax, tmin=obs_tmin, p90_tmax=p90tmax_thres, p90_tmin=p90tmin_thres,
                                min_len=MIN_DAYS, type=TYPE_HEAT)
    
    metrics_obs_all[iyear] = metrics_from_mask(lat=lat_glob, lon=lon_glob, mask=hw_mask_obs, anom=anom_obs)
    del obs_tmax, obs_tmin, anom_obs
    hw_mask_obs = hw_mask_obs.astype(bool)

    # --- Step 2: get the forecasts
    pred_og_tmax = np.full((N_init, NY, NX), np.nan, dtype=np.float32)
    pred_og_tmin = np.full((N_init, NY, NX), np.nan, dtype=np.float32)

    pred_corr_tmax = np.full((N_init, NY, NX), np.nan, dtype=np.float32)
    pred_corr_tmin = np.full((N_init, NY, NX), np.nan, dtype=np.float32)

    # --- loop over initialization dates
    for i, init in enumerate(init_dates):
    
        # load prediction outputs for original ERA5 group
        folder_og = FORECAST_OG_DIR / f"{init:%Y-%m-%dT00Z}"
        if folder_og.exists():
            fh_og = folder_og / f"pred_{init:%Y-%m-%dT00Z}_{lead_day*24:03d}.nc"
            
            if not fh_og.exists():
                print(f"[Skip] did not find {fh_og}", flush=True)
                continue

            if i  > (N_init-1):
                print(f"[Skip] Index {i} is out of range for N_init={N_init}", flush=True)
                continue

            ds = xr.open_dataset(fh_og)
            pred_og_tmax[i, :, :] = ( ds["tmax"].isel(time=0).values - 273.15 ).astype(np.float32) # convert from K to °C
            pred_og_tmin[i, :, :] = ( ds["tmin"].isel(time=0).values - 273.15 ).astype(np.float32)
            lat = ds["latitude"].values.astype(np.float32)
            lon = ds["longitude"].values.astype(np.float32)
            ds.close()
        

        # load prediction outputs for bias-corrected ERA5 group
        folder_corr = FORECAST_CORR_DIR / f"{init:%Y-%m-%dT00Z}"
        if folder_corr.exists():
            fh_corr = folder_corr / f"pred_{init:%Y-%m-%dT00Z}_{lead_day*24:03d}.nc"
            
            if not fh_corr.exists():
                print(f"[Skip] did not find {fh_corr}", flush=True)
                continue

            if i  > (N_init-1):
                print(f"[Skip] Index {i} is out of range for N_init={N_init}", flush=True)
                continue

            ds = xr.open_dataset(fh_corr)
            pred_corr_tmax[i, :, :] = ( ds["tmax"].isel(time=0).values - 273.15 ).astype(np.float32) # convert from K to °C
            pred_corr_tmin[i, :, :] = ( ds["tmin"].isel(time=0).values - 273.15 ).astype(np.float32)
            lat = ds["latitude"].values.astype(np.float32)
            lon = ds["longitude"].values.astype(np.float32)
            ds.close()

    
    hw_mask_og, anom_og = detect_days(time=obs_time, lat = lat_glob, tmax=pred_og_tmax, tmin=pred_og_tmin, p90_tmax=p90tmax_thres, p90_tmin=p90tmin_thres,
                                min_len=MIN_DAYS, type=TYPE_HEAT)    
    metrics_og_all[iyear] = metrics_from_mask(lat=lat_glob, lon=lon_glob, mask=hw_mask_og, anom=anom_og)
    del pred_og_tmax, pred_og_tmin, anom_og            
    hw_mask_og = hw_mask_og.astype(bool)
    TP_og_all[iyear] = np.sum(hw_mask_og & hw_mask_obs, axis=0)
    FP_og_all[iyear] = np.sum(hw_mask_og & ~hw_mask_obs, axis=0)
    TN_og_all[iyear] = np.sum(~hw_mask_og & ~hw_mask_obs, axis=0)
    FN_og_all[iyear] = np.sum(~hw_mask_og & hw_mask_obs, axis=0)


    hw_mask_corr, anom_corr = detect_days(time=obs_time, lat = lat_glob, tmax=pred_corr_tmax, tmin=pred_corr_tmin, p90_tmax=p90tmax_thres, p90_tmin=p90tmin_thres,
                                min_len=MIN_DAYS, type=TYPE_HEAT)    
    metrics_corr_all[iyear] = metrics_from_mask(lat=lat_glob, lon=lon_glob, mask=hw_mask_corr, anom=anom_corr)
    del pred_corr_tmax, pred_corr_tmin, anom_corr
    hw_mask_corr = hw_mask_corr.astype(bool)
    TP_corr_all[iyear] = np.sum(hw_mask_corr & hw_mask_obs, axis=0)
    FP_corr_all[iyear] = np.sum(hw_mask_corr & ~hw_mask_obs, axis=0)
    TN_corr_all[iyear] = np.sum(~hw_mask_corr & ~hw_mask_obs, axis=0)
    FN_corr_all[iyear] = np.sum(~hw_mask_corr & hw_mask_obs, axis=0)


results = {
    "metrics_obs": metrics_obs_all,
    "metrics_og": metrics_og_all,
    "metrics_corr": metrics_corr_all,

    "TP_og": TP_og_all,
    "FP_og": FP_og_all,
    "TN_og": TN_og_all,
    "FN_og": FN_og_all,

    "TP_corr": TP_corr_all,
    "FP_corr": FP_corr_all,
    "TN_corr": TN_corr_all,
    "FN_corr": FN_corr_all,
}

with open(OUT_DIR/f"CREDIT_heatwave_type{TYPE_HEAT}_lead{lead_day:02d}.pkl", "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

## later load back
#with open(f"CREDIT_forecast_heatwave_lead{lead_day:02d}.pkl", "rb") as f:
#    results = pickle.load(f)