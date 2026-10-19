# %% [markdown]
# #### Pair GHCN-d observations with ERA5 spatially
# Prerequisite: GHCN-d observations are already converted to UTC time coordinate
#
# Main steps when looping over all year / month / date
# 1. extract TMIN_obs and TMAX_obs from GHCNd observations converted to UTC time
# 2. compute TMIN_era and TMAX_era from ERA5 hourly reanalysis data of 2m air temperatures
# 3. adjust TMIN_obs and TMAX_obs to be on same lat & lon coordinates as ERA5 reanalysis data
# 4. perform residual kriging on daily TMAX_obs and TMIN_obs distribution across the globe
# 5. compare kriged TMAX_obs with original TMAX_era, and kriged TMIN_obs with original TMIN_era

# %%
# load the packages
import numpy as np
import pandas as pd
import calendar
import h5py
import gstools as gs
import netCDF4 as nc
from pathlib import Path
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer

# install gstools using "pip install gstools"
import argparse

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Process subset of years")
parser.add_argument("--start", type=int, required=True, help="Start year index (0-based)")
parser.add_argument("--end", type=int, required=True, help="End year index(exclusive)")
args = parser.parse_args()

start_idx = args.start
end_idx = args.end
#print(f"Processing years {1980+start_idx} to {1980+end_idx-1}")

# %%
# set the filepath and parameters

dir_ghcn_UTC = Path("/ourdisk/hpc/ai2es/yxie/T_ghcnd_UTC/")
dir_era = Path("/ourdisk/hpc/ai2es/yxie/ERA5")
dir_output = Path("/scratch/yxie/krig_data/")

#FIRST_YEAR = int(1980 + start_idx)
#LAST_YEAR = int(1980 + end_idx - 1)
#FIRST_MONTH = int(7)
#LAST_MONTH = int(12)

FIRST_YEAR = 2025
LAST_YEAR = 2025
FIRST_MONTH = int(start_idx)
LAST_MONTH = int(end_idx - 1)

# %%
def ndays_in_months(year, month):
    return calendar.monthrange(year, month)[1]


# %%
def linear_taper(r):
    """
    linear taper weight for blending
    linearly decreasing weights 1->0 as r goes 0->1
    """
    rmx = np.nanmax(r) + 1e-8
    return np.clip(1.0 - r / rmx, 0.0, 1.0)


# %%
def load_ghcn_utc_month(year, month):
    fname = dir_ghcn_UTC / f"UTC_{year}{month:02d}.parquet"
    df = pd.read_parquet(fname)
    # drop missing or invalid elevation
    mask = np.isfinite(df["ELEV"]) & (df["ELEV"] != -999.9)
    df_clean = df.loc[mask].copy()
    # make sure the DATE values are int
    df_clean["DATE"] = df_clean["DATE"].astype(int)
    del df
    return df_clean


# %%
def load_era5_month(year, month):
    """
    Open a monthly ERA5 file and return (time[og], lat[og], lon[-180 t0 180], t2m[degC]).
    """
    fname = dir_era / f"ERA5reanalysis_T2m_hourly_{year}{month:02d}.nc"
    with nc.Dataset(fname, "r") as ds:
        t = ds.variables["valid_time"][:]  # shape (744, ), seconds since 1970-01-01
        lat = ds.variables["latitude"][:]  # shape (721, ), degrees north: 90: -90 : -0.25
        lon0 = ds.variables["longitude"][:]  # shape (1440, ), degrees east: 0: 359.75 :0.25
        t2m0 = ds.variables["t2m"][:, :, :]  # shape (744, 721, 1440), 2-m air temperature, units: K

    # --- convert lon (0-360) to lon (-180-180) --- #
    idlon1 = np.where((lon0 >= 0) & (lon0 < 180))[0]
    idlon2 = np.where((lon0 >= 180) & (lon0 < 360))[0]
    lon = np.concatenate((lon0[idlon2] - 360.0, lon0[idlon1]), axis=0)
    t2m = np.concatenate((t2m0[:, :, idlon2], t2m0[:, :, idlon1]), axis=2)
    # convert Kelvin to degC
    t2m = t2m - 273.15
    del lon0, t2m0

    return t, lat, lon, t2m


# %%
# set the basis used for residual kriging
def design(lat, lon, elev_km):
    X = np.column_stack(
        [
            np.ones_like(lat),
            lat,
            lon,
            elev_km,
            lat**2,
            lon**2,
            lat * lon,
            lat * elev_km,
            lon * elev_km,
        ]
    )
    return X


# %%
def equal_count_bins(d, n_bins=25):
    # d vector of pairwise distances from semivariance calculation
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(d, qs)
    edges[0] -= 1e-6  # to include the minimum value
    return np.unique(edges)


# %%
# Haversine / Great circle distance in kilometers
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# %%
# Helper for Azimuthal-Equidistant (AEQD) projection per tile
# within a small tile, we can not directly use lat-lon degrees
# because one degree of longitude varies across latitudes, e.g., shrink near the poles
# while kriging expect consistent metric (kilometers or meters) distances
def to_aeqd(lat0, lon0, lat, lon):
    """Project lat/lon (deg) to local AEQD meters centered at (lat0, lon0)"""
    crs_geod = CRS.from_epsg(4326)
    crs_loc = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    tf = Transformer.from_crs(crs_geod, crs_loc, always_xy=True)
    x_m, y_m = tf.transform(lon, lat)
    return np.asarray(x_m) / 1000.0, np.asarray(y_m) / 1000.0


# %%
# ----- implement fast KDTree variogram sampling -----
def compute_variogram_kdtree(
    lats, lons, vals, k=600, R=6371.0, max_dist_km=None, n_bins=50, min_pairs_per_bin=20
):
    """
    Compute experimental variogram using k-nearest-neighbor sampling (unique unordered pairs) and
    return both the raw pairwise values and the binned (distance, semi-variance) averages

    Parameters
    ----------
    lats, lons: array-like
        Station coordinates in degrees
    vals: array-like
        Scalar variable at each station (e.g., temperature, errors in temperature)
    k: int, optional
        Numer of nearest neighbors per station (default=600)
    R: float, optional
        Earth's radius in km
    max_dist_km: float or None
        Optional cutoff to ignore long-distance pairs
    n_bins: int, optional
        Number of distance bins for averaging (default=50)
    min_pairs_per_bin: int, optional
        Minimum number of pairs needed fo rbin to be kept

    Returns
    ----------
    pairs_d, pairs_g : np.ndarray
        All unique pairwise distances and semivariances
    bin_centers, gamma_mean, gamma_std: np.ndarray
        Binned variogram statistics (distance center, mean gamma, std gamma)
    """
    # ----- 0: sanity check - keep valid values -----
    n = len(vals)
    mask = np.isfinite(vals)
    lats, lons, vals = lats[mask], lons[mask], vals[mask]

    # ----- 1: build 3D coordinates on unit sphere -----
    # convert (lat, lon) to 3D unit vectors for geodesic distance (great circle metric)
    lat_r, lon_r = np.deg2rad(lats), np.deg2rad(lons)
    xyz = np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )

    # ----- 2: kNN search -----
    tree = cKDTree(xyz)
    dist, idx = tree.query(xyz, k=k + 1)
    # drop the station itself
    dist, idx = dist[:, 1:], idx[:, 1:]

    # ----- 3: flatten & remove duplicate pairs -----
    I = np.repeat(np.arange(n), k)
    J = idx.ravel()
    D_chord = dist.ravel()

    D_km = 2 * R * np.arcsin(np.clip(D_chord / 2.0, 0, 1))
    mask = I < J
    I, J, D_km = I[mask], J[mask], D_km[mask]

    if max_dist_km is not None:
        sel = D_km <= max_dist_km
        I, J, D_km = I[sel], J[sel], D_km[sel]

    # ----- 4: compute semivariance -----
    pairs_g = 0.5 * (vals[I] - vals[J]) ** 2
    pairs_d = D_km

    # ----- 5: Bin distance -----
    # choose equal-distance binning or equal-count binning
    # bins = np.linspace(0, pairs_d.max(), n_bins+1)
    bins = equal_count_bins(pairs_d, n_bins=n_bins)

    # return the indices of the bins to which each value in input array belongs
    bin_idx = np.digitize(pairs_d, bins) - 1
    bin_centers, gamma_mean, gamma_std = [], [], []

    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) >= min_pairs_per_bin:
            gamma_mean.append(np.mean(pairs_g[mask]))
            gamma_std.append(np.std(pairs_g[mask]))
            bin_centers.append(0.5 * (bins[b] + bins[b + 1]))

    return (pairs_d, pairs_g, np.array(bin_centers), np.array(gamma_mean), np.array(gamma_std))


# %%
def fit_daily_variogram(
    df_ghcn_day,
    element: str,
    year: int,
    month: int,
    iday: int,
    K=800,  # number of nearest neighbors
    MAX_DIST_KM=600,  # maximum distance in units of kilometers
    N_BINS=25,  # number of bins for variogram
):
    """
    Fit a daily variogram model for one element (daily TMAX or TMIN)
    and return a gs.Stable model
    """
    assert element in ("TMAX_UTC", "TMIN_UTC")
    params = []

    YYYYMMDD = int(year * 10000 + month * 100 + iday)
    df_day = df_ghcn_day.copy()
    # remove missing values / infinite values
    df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_day.dropna(subset=[element], inplace=True)

    # build residuals: detrend vs lat, lon, and elev_km
    lats = df_day["LAT"].to_numpy()
    lons = df_day["LON"].to_numpy()
    elevs_km = df_day["ELEV"].to_numpy() / 1000.0
    vals = df_day[element].to_numpy()
    X = design(lat=lats, lon=lons, elev_km=elevs_km)
    beta = np.linalg.lstsq(X, vals, rcond=None)[0]
    trend = X @ beta
    resid = vals - trend
    # compute variogram for this day
    dist, gamma, bin_cen, gamma_mean, gamma_std = compute_variogram_kdtree(
        lats=lats, lons=lons, vals=resid, k=K, max_dist_km=MAX_DIST_KM, n_bins=N_BINS
    )
    # clean NaNs
    m = np.isfinite(bin_cen) & np.isfinite(gamma_mean)
    bin_cen = bin_cen[m]
    gamma_mean = gamma_mean[m]
    # initial guess for model parameters
    init_guess = {
        "nugget": 1.5,
        "len_scale": 100.0,  # ~ correlation scale in km
        "var": float(np.nanmax(gamma_mean)) if np.isfinite(gamma_mean).any() else 2.0,
    }
    model_day = gs.Stable(dim=2)
    try:
        model_day.fit_variogram(
            x_data=bin_cen,
            y_data=gamma_mean,
            nugget=True,
            len_scale=True,
            sill=True,
            init_guess=init_guess,
            max_eval=20000,
        )
        print(f"[INFO] Variogram model fit success for {element} on {YYYYMMDD}")
    except Exception as e:
        print(f"[WARN] Variogram model fit failed for {element} on {YYYYMMDD} : {e}")
        # choose a generic fallback
        return gs.Exponential(dim=2, var=float(np.nanmax(gamma_mean)), len_scale=100)

    return model_day


# %%
def select_stratified_neighbors(
    xs, ys, keep, NMAX, shell_edges=(0.0, 100.0, 200.0, 300.0, 400, 500, 600.0), rng=None
):
    """
    Stratified sampling by distance shells.

    Parameters
    ----------
    xs, ys : 1D arrays
        Station coordinates in AEQD (km), same length as full station list.
    keep : 1D int array
        Indices of candidate stations inside BUF_KM / SEARCH_KM.
    NMAX : int or None
        Maximum number of neighbors to keep. If None, returns keep unchanged.
    shell_edges : tuple of float
        Distance bounds (km).
    rng : np.random.Generator or None
        RNG for reproducible sampling.

    Returns
    -------
    keep_strat : 1D int array
        Subset of `keep` with at most NMAX neighbors, stratified by distance.
    """
    if (NMAX is None) or (keep.size <= NMAX):
        return keep

    if rng is None:
        rng = np.random.default_rng()

    # distance from tile center in AEQD km
    d = np.sqrt(xs[keep] ** 2 + ys[keep] ** 2)

    shell_edges = np.asarray(shell_edges, float)
    n_shells = len(shell_edges) - 1
    if n_shells <= 0:
        # fallback to nearest NMAX
        d2c = xs[keep] ** 2 + ys[keep] ** 2
        order = np.argsort(d2c)
        return keep[order[:NMAX]]

    # equal quota per shell (you can tweak this)
    base_quota = NMAX // n_shells
    leftover = NMAX - base_quota * n_shells

    selected = []
    used = 0

    for s in range(n_shells):
        r0, r1 = shell_edges[s], shell_edges[s + 1]

        mask_shell = (d >= r0) & (d < r1)
        idx_shell = np.where(mask_shell)[0]  # positions within keep

        if idx_shell.size == 0:
            continue

        # distribute leftover 1-by-1 to inner shells first
        quota = base_quota + (1 if s < leftover else 0)
        take = min(quota, idx_shell.size)

        if take > 0:
            chosen_local = rng.choice(idx_shell, size=take, replace=False)
            selected.append(keep[chosen_local])
            used += take

    if used == 0:
        # If something went wrong, fall back to nearest NMAX
        d2c = xs[keep] ** 2 + ys[keep] ** 2
        order = np.argsort(d2c)
        return keep[order[:NMAX]]

    keep_strat = np.unique(np.concatenate(selected))

    # If we still have room (due to empty shells), top up with closest remaining
    if keep_strat.size < NMAX:
        remaining_mask = ~np.isin(keep, keep_strat)
        remaining = keep[remaining_mask]
        if remaining.size > 0:
            d2c_rest = xs[remaining] ** 2 + ys[remaining] ** 2
            n_more = min(NMAX - keep_strat.size, remaining.size)
            more = remaining[np.argsort(d2c_rest)[:n_more]]
            keep_strat = np.unique(np.concatenate([keep_strat, more]))

    return keep_strat


# %%
def load_orography(fname):
    """
    Return (lat, lon, orography_meters) or None if not available
    """
    if fname is None or not fname.exists():
        return None

    with nc.Dataset(fname, "r") as ds:
        lat = ds.variables["latitude"][:]  # latitude, shape (721, ), degrees north: 90: -90 : -0.25
        lon = ds.variables["longitude"][
            :
        ]  # longitude, shape (1440, ), degrees east: 0: 359.75 :0.25
        z = ds.variables["z"][0, :, :]  # geopotential, shape (1, 721, 1440), units: m**2 s**-2
        G = 9.80665  # gravitational acceleration, units: m s**-2
        oro = z / G
    return (np.array(lat), np.array(lon), np.array(oro))


OROG = load_orography(fname=dir_era / "ERA5reanalysis_geopotential_sfc.nc")


def load_landmask(fname):
    """
    Return (lat, lon, land_sea_mask) or None if not available
    """
    if fname is None or not fname.exists():
        return None
    with nc.Dataset(fname, "r") as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        lsm = ds.variables["lsm"][0, :, :]  # land binary mask;s 0: ocean; 1: land
    return (np.array(lat), np.array(lon), np.array(lsm))


LSMASK = load_landmask(fname=dir_era / "ERA5reanalysis_landseamask.nc")

if OROG:
    lat_era0, lon_era0, elev_era0 = OROG
    if LSMASK:
        _, _, lsm_era0 = LSMASK

# --- convert lon (0-360) to lon (-180-180) --- #
idlon1 = np.where((lon_era0 >= 0) & (lon_era0 < 180))[0]
idlon2 = np.where((lon_era0 >= 180) & (lon_era0 < 360))[0]
lat_era = lat_era0
lon_era = np.concatenate((lon_era0[idlon2] - 360.0, lon_era0[idlon1]), axis=0)
# convert meters to kilometers
elev_era = np.concatenate((elev_era0[:, idlon2], elev_era0[:, idlon1]), axis=1) / 1000.0
lsm_era = np.concatenate((lsm_era0[:, idlon2], lsm_era0[:, idlon1]), axis=1)


# %%
# seamless version - kriging by footprint
if __name__ == "__main__":
    # ----- global kriging setting parameters -----
    K = 800
    MAX_DIST_KM = 600
    N_BINS = 25

    # ----- kriging tile parameters -----
    TILE_STEP_DEG = 5.0  # space between tile centers
    TILE_RADIUS_KM = 400.0  # kriging footprint radius (controls overlap)
    STA_BUFFER_KM = 800.0  # stations included if located within this distance
    SEARCH_KM = 600.0  # max distance for kriging neighbors selection
    NMAX = 360  # cap number of neighbors (to prevent sigular matrices)

    for iyear in range(FIRST_YEAR, LAST_YEAR + 1):
        for imon in range(FIRST_MONTH, LAST_MONTH + 1):
            # ========== load GHCN obs. at UTC time ========== #
            df_ghcn_mon = load_ghcn_utc_month(year=iyear, month=imon)
            # load ERA5 reanalysis for the month
            t_era, lat_era, lon_era, t2m_era = load_era5_month(year=iyear, month=imon)
            # number of days in this month
            nd = ndays_in_months(year=iyear, month=imon)

            # ========== allocate monthly output arrays ========= #
            nlat = lat_era.size
            nlon = lon_era.size
            # --- for GHCNd obs.
            pred_tmax_tot = np.full((nd, nlat, nlon), np.nan, dtype=float)
            std_krig_tmax_tot = np.full_like(pred_tmax_tot, np.nan)
            pred_tmin_tot = np.full((nd, nlat, nlon), np.nan, dtype=float)
            std_krig_tmin_tot = np.full_like(pred_tmin_tot, np.nan)
            # --- for ERA5 reanalysis
            era_tmax_tot = np.full((nd, nlat, nlon), np.nan, dtype=float)
            era_tmin_tot = np.full((nd, nlat, nlon), np.nan, dtype=float)

            # precompute ERA5 mesh
            LLON_full, LLAT_full = np.meshgrid(lon_era, lat_era)
            # create tile centers on a regular 5-deg grid
            lat_centers = np.arange(lat_era.max(), lat_era.min() - TILE_STEP_DEG, -TILE_STEP_DEG)
            lon_centers = np.arange(lon_era.min(), lon_era.max() + TILE_STEP_DEG, TILE_STEP_DEG)

            flag_day = False
            for iday in range(1, nd + 1):
                try:
                    # =============================== #
                    #  load GHCN-d and ERA5 data
                    # =============================== #
                    # GHCN-d daily TMAX and TMIN
                    YYYYMMDD = int(iyear * 10000 + imon * 100 + iday)
                    TMAX_obs_day = df_ghcn_mon.loc[
                        df_ghcn_mon["DATE"] == YYYYMMDD, ["ID", "LAT", "LON", "ELEV", "TMAX_UTC"]
                    ].copy()
                    TMIN_obs_day = df_ghcn_mon.loc[
                        df_ghcn_mon["DATE"] == YYYYMMDD, ["ID", "LAT", "LON", "ELEV", "TMIN_UTC"]
                    ].copy()
                    # remove missing values / infinite values
                    TMAX_obs_day.replace([np.inf, -np.inf], np.nan, inplace=True)
                    TMAX_obs_day.dropna(subset=["TMAX_UTC"], inplace=True)
                    TMIN_obs_day.replace([np.inf, -np.inf], np.nan, inplace=True)
                    TMIN_obs_day.dropna(subset=["TMIN_UTC"], inplace=True)
                    # ERA5 daily TMAX and TMIN
                    mask_iday = np.where(
                        ((t_era - t_era[0]) >= (iday - 1) * 24 * 60 * 60)
                        & ((t_era - t_era[0]) < iday * 24 * 60 * 60)
                    )[0]
                    TMAX_era_day = np.nanmax(t2m_era[mask_iday, :, :], axis=0)
                    TMIN_era_day = np.nanmin(t2m_era[mask_iday, :, :], axis=0)

                    # ================================= #
                    # perform residual kriging on GHCN-d
                    # ================================= #
                    # remove basis trend on coordinates: lat, lon, elevation
                    lats_tmax = TMAX_obs_day["LAT"].to_numpy()
                    lons_tmax = TMAX_obs_day["LON"].to_numpy()
                    elevs_tmax = TMAX_obs_day["ELEV"].to_numpy() / 1000.0
                    tmax = TMAX_obs_day["TMAX_UTC"].to_numpy()

                    lats_tmin = TMIN_obs_day["LAT"].to_numpy()
                    lons_tmin = TMIN_obs_day["LON"].to_numpy()
                    elevs_tmin = TMIN_obs_day["ELEV"].to_numpy() / 1000.0
                    tmin = TMIN_obs_day["TMIN_UTC"].to_numpy()

                    X_tmax = design(lat=lats_tmax, lon=lons_tmax, elev_km=elevs_tmax)
                    beta_tmax = np.linalg.lstsq(X_tmax, tmax, rcond=None)[0]
                    trend_tmax = X_tmax @ beta_tmax
                    resid_tmax = tmax - trend_tmax

                    X_tmin = design(lat=lats_tmin, lon=lons_tmin, elev_km=elevs_tmin)
                    beta_tmin = np.linalg.lstsq(X_tmin, tmin, rcond=None)[0]
                    trend_tmin = X_tmin @ beta_tmin
                    resid_tmin = tmin - trend_tmin

                    # ========== fit daily variogram models ========== #
                    fit_model_daily_tmax = fit_daily_variogram(
                        df_ghcn_day=TMAX_obs_day,
                        element="TMAX_UTC",
                        year=iyear,
                        month=imon,
                        iday=iday,
                        K=K,
                        MAX_DIST_KM=MAX_DIST_KM,
                        N_BINS=N_BINS,
                    )
                    fit_model_daily_tmin = fit_daily_variogram(
                        df_ghcn_day=TMIN_obs_day,
                        element="TMIN_UTC",
                        year=iyear,
                        month=imon,
                        iday=iday,
                        K=K,
                        MAX_DIST_KM=MAX_DIST_KM,
                        N_BINS=N_BINS,
                    )

                    # ============================== #
                    #  allocate global accumulators  #
                    # ============================== #
                    pred_sum_tmax = np.zeros((nlat, nlon))
                    pred_sum_tmin = np.zeros((nlat, nlon))
                    var_sum_tmax = np.zeros((nlat, nlon))
                    var_sum_tmin = np.zeros((nlat, nlon))
                    w_sum_tmax = np.zeros((nlat, nlon))
                    w_sum_tmin = np.zeros((nlat, nlon))

                    # ===================================== #
                    #   loop over all tile (tile centers)   #
                    # ===================================== #
                    for lat_c in lat_centers:
                        for lon_c in lon_centers:
                            # ERA5 grid points within tile footprint (circle)
                            dist_grid = haversine_km(LLAT_full, LLON_full, lat_c, lon_c)
                            mask = dist_grid <= TILE_RADIUS_KM
                            if not mask.any():
                                continue
                            
                            # weights for blending
                            rho = dist_grid[mask] / TILE_RADIUS_KM
                            # w = cosine_taper(rho)
                            w = linear_taper(rho)
                            
                            # ERA5 subgrid
                            lat_sub = LLAT_full[mask]
                            lon_sub = LLON_full[mask]
                            elev_sub = elev_era[mask]

                            # ----- station subset for TMAX ----- #
                            dist_sta = haversine_km(lats_tmax, lons_tmax, lat_c, lon_c)
                            sta_mask_tmax = dist_sta <= STA_BUFFER_KM
                            if sta_mask_tmax.sum() >= 4:
                                lat_loc = lats_tmax[sta_mask_tmax]
                                lon_loc = lons_tmax[sta_mask_tmax]
                                res_loc = resid_tmax[sta_mask_tmax]
                                # project to AEQD centered on tile
                                xs, ys = to_aeqd(lat_c, lon_c, lat_loc, lon_loc)
                                xg, yg = to_aeqd(lat_c, lon_c, lat_sub, lon_sub)
                                # neighbor search structure
                                kdt = cKDTree(np.c_[xs, ys])
                                idxs = kdt.query_ball_point(np.c_[xg, yg], r=SEARCH_KM)
                                keep = (
                                    np.unique(
                                        np.concatenate(
                                            [np.array(ix, int) for ix in idxs if len(ix) > 0]
                                        )
                                    )
                                    if len(idxs) > 0
                                    else np.array([], int)
                                )
                                if keep.size >= 4:
                                    # constrain maximum neighbor size
                                    if NMAX and keep.size > NMAX:
                                        # d2c = xs[keep]**2 + ys[keep]**2
                                        # keep = keep[np.argsort(d2c)[:NMAX]]
                                        shell_edges = (
                                            0.0,
                                            100.0,
                                            200.0,
                                            300.0,
                                            400.0,
                                            500.0,
                                            SEARCH_KM,
                                        )
                                        keep = select_stratified_neighbors(
                                            xs, ys, keep, NMAX=NMAX, shell_edges=shell_edges
                                        )
                                    pos = (xs[keep], ys[keep])
                                    val = res_loc[keep]
                                    # ordinary kriging of residuals over tile grid
                                    ok = gs.krige.Ordinary(
                                        fit_model_daily_tmax,
                                        cond_pos=pos,
                                        cond_val=val,
                                        exact=False,
                                    )
                                    res_pred, res_var = ok((xg, yg), return_var=True)
                                    # add back trend
                                    Xg = design(lat_sub, lon_sub, elev_sub)
                                    trend_grid = Xg @ beta_tmax
                                    pred_sub = trend_grid + res_pred
                                    # accumulate
                                    pred_sum_tmax[mask] += w * pred_sub
                                    var_sum_tmax[mask] += (w**2) * res_var
                                    w_sum_tmax[mask] += w

                            # ----- station subset for TMIN ----- #
                            dist_sta = haversine_km(lats_tmin, lons_tmin, lat_c, lon_c)
                            sta_mask_tmin = dist_sta <= STA_BUFFER_KM
                            if sta_mask_tmin.sum() >= 4:
                                lat_loc = lats_tmin[sta_mask_tmin]
                                lon_loc = lons_tmin[sta_mask_tmin]
                                res_loc = resid_tmin[sta_mask_tmin]
                                # project to AEQD centered on tile
                                xs, ys = to_aeqd(lat_c, lon_c, lat_loc, lon_loc)
                                xg, yg = to_aeqd(lat_c, lon_c, lat_sub, lon_sub)
                                # neighbor search structure
                                kdt = cKDTree(np.c_[xs, ys])
                                idxs = kdt.query_ball_point(np.c_[xg, yg], r=SEARCH_KM)
                                keep = (
                                    np.unique(
                                        np.concatenate(
                                            [np.array(ix, int) for ix in idxs if len(ix) > 0]
                                        )
                                    )
                                    if len(idxs) > 0
                                    else np.array([], int)
                                )
                                if keep.size >= 4:
                                    # constrain maximum neighbor size
                                    if NMAX and keep.size > NMAX:
                                        # d2c = xs[keep]**2 + ys[keep]**2
                                        # keep = keep[np.argsort(d2c)[:NMAX]]
                                        shell_edges = (
                                            0.0,
                                            100.0,
                                            200.0,
                                            300.0,
                                            400.0,
                                            500.0,
                                            SEARCH_KM,
                                        )
                                        keep = select_stratified_neighbors(
                                            xs, ys, keep, NMAX=NMAX, shell_edges=shell_edges
                                        )
                                    pos = (xs[keep], ys[keep])
                                    val = res_loc[keep]
                                    # ordinary kriging of residuals over tile grid
                                    ok = gs.krige.Ordinary(
                                        fit_model_daily_tmin,
                                        cond_pos=pos,
                                        cond_val=val,
                                        exact=False,
                                    )
                                    res_pred, res_var = ok((xg, yg), return_var=True)
                                    # add back trend
                                    Xg = design(lat_sub, lon_sub, elev_sub)
                                    trend_grid = Xg @ beta_tmin
                                    pred_sub = trend_grid + res_pred
                                    # accumulate
                                    pred_sum_tmin[mask] += w * pred_sub
                                    var_sum_tmin[mask] += (w**2) * res_var
                                    w_sum_tmin[mask] += w

                    # wrap up blended predictions for this day
                    pred_tmax_day = np.where(w_sum_tmax > 0, pred_sum_tmax / w_sum_tmax, np.nan)
                    var_tmax_day = np.where(w_sum_tmax > 0, var_sum_tmax / (w_sum_tmax**2), np.nan)
                    pred_tmin_day = np.where(w_sum_tmin > 0, pred_sum_tmin / w_sum_tmin, np.nan)
                    var_tmin_day = np.where(w_sum_tmin > 0, var_sum_tmin / (w_sum_tmin**2), np.nan)

                    # apply land-sea mask
                    mask_lsm = lsm_era > 0.5
                    pred_tmax_day[~mask_lsm] = np.nan
                    var_tmax_day[~mask_lsm] = np.nan
                    pred_tmin_day[~mask_lsm] = np.nan
                    var_tmin_day[~mask_lsm] = np.nan

                    # save the day
                    pred_tmax_tot[iday - 1, :, :] = pred_tmax_day
                    std_krig_tmax_tot[iday - 1, :, :] = np.sqrt(var_tmax_day)
                    pred_tmin_tot[iday - 1, :, :] = pred_tmin_day
                    std_krig_tmin_tot[iday - 1, :, :] = np.sqrt(var_tmin_day)
                    era_tmax_tot[iday - 1, :, :] = TMAX_era_day
                    era_tmin_tot[iday - 1, :, :] = TMIN_era_day

                    flag_day = True

                except Exception as e:
                    print(f"[Fail] {iyear}-{imon:02d}-{iday:02d} : {e}")

            # save the month's data
            # lat, lon, elev,
            # era_tmax, kriged tmax values and its uncertainty standard deviation
            # era_tmin, kriged tmin values and its uncertainty standard deviation
            # write each month's paired ERA5 data with the same format
            if flag_day:
                f_h5_out = dir_output / f"tmnmx_pair_{iyear}{imon:02d}.h5"
                with h5py.File(f_h5_out, "w") as h5out:
                    # preserve ID dtype (bytes vs unicode) to make comparison easier
                    h5out.create_dataset(
                        "lat", data=lat_era, dtype=lat_era.dtype, compression="gzip", shuffle=True
                    )
                    h5out.create_dataset(
                        "lon", data=lon_era, dtype=lon_era.dtype, compression="gzip", shuffle=True
                    )
                    h5out.create_dataset(
                        "elev",
                        data=elev_era,
                        dtype=elev_era.dtype,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "era_tmax",
                        data=era_tmax_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "era_tmin",
                        data=era_tmin_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "ghcn_tmax",
                        data=pred_tmax_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "std_tmax",
                        data=std_krig_tmax_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "ghcn_tmin",
                        data=pred_tmin_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )
                    h5out.create_dataset(
                        "std_tmin",
                        data=std_krig_tmin_tot,
                        dtype=np.float32,
                        compression="gzip",
                        shuffle=True,
                    )

                    # add attributes for information
                    h5out.attrs["year"] = iyear
                    h5out.attrs["month"] = imon
                    h5out.attrs["units_temp"] = np.bytes_("deg C")
                    h5out.attrs["units_elev"] = np.bytes_("km")

                print(f"[Success] {iyear}{imon:02d}")
