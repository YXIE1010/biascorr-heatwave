# %% [markdown]
# #### Pair GHCN-d observations with ERA5 temporally
# Two conversions to be performed:
# 1. From local reporting time to local solar time
# 2. From local solar time to UTC coordinate

# %%
# load modules
from pathlib import Path
import calendar
import pandas as pd
import numpy as np
import argparse

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Process subset of years")
parser.add_argument("--start", type=int, required=True, help="Start year index (0-based)")
parser.add_argument("--end", type=int, required=True, help="End year index(exclusive)")
args = parser.parse_args()

start_idx = args.start
end_idx = args.end
print(f"Processing years {1980+start_idx} to {1980+end_idx}")


# %%
# set folder path and parameters
fpath = Path("/ourdisk/hpc/ai2es/yxie")
obs_by_year = fpath / "ghcnd_by_year"
dir_output = Path("/scratch/yxie")


# %%
def read_one_year_subset(year, date_list):
    """
    Efficiently read only specific dates from YYYY.csv using chunked loading
    date_list = list of YYYYMMDD strings to keep
    """
    fp = obs_by_year / f"{year}.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    keep_dates = set(date_list)
    chunks = []

    for chunk in pd.read_csv(
        fp,
        header=None,
        dtype=str,
        names=["ID", "DATE", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBSTIME"],
        chunksize=200_000,
    ):
        sub = chunk[chunk["DATE"].isin(keep_dates)]
        if not sub.empty:
            chunks.append(sub)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=["ID", "DATE", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBSTIME"]
        )


# %%
def extract_month_window(year, month):
    """
    Load only the needed dates:
     - prev month last day
     - this month all days
     - next month first day
    Without loading the full YYYY.csv files
    """
    # current month range
    yyyy = str(year)
    mm = f"{month:02d}"
    ndays = calendar.monthrange(year, month)[1]
    month_dates = [f"{yyyy}{mm}{d:02d}" for d in range(1, ndays + 1)]

    # previous day
    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1

    prev_ndays = calendar.monthrange(prev_year, prev_month)[1]
    prev_date = f"{prev_year}{prev_month:02d}{prev_ndays:02d}"

    # next day
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    next_date = f"{next_year}{next_month:02d}01"

    # build lists per-year
    need_prev = [prev_date]
    need_this = month_dates
    need_next = [next_date]

    df_prev = read_one_year_subset(prev_year, need_prev)
    df_this = read_one_year_subset(year, need_this)
    df_next = read_one_year_subset(next_year, need_next)

    df_concat = pd.concat([df_prev, df_this, df_next], ignore_index=True)
    return df_concat


# %%
def extract_TMAX_TMIN_records(df):
    """
    - Keep only TMAX and TMIN
    - Skip rows where QFLAG is non-blank
    - Convert value to numeric (divide by 10 for degC)
    - Return separate wide-format DataFrame
    """
    # keep TMAX/TMIN only
    df2 = df[df["ELEMENT"].isin(["TMAX", "TMIN"])].copy()
    # keep only blank quality flag
    df2 = df2[df2["QFLAG"].isna() | (df2["QFLAG"] == "")]
    # convert temperature values to numeric
    df2["VALUE"] = pd.to_numeric(df2["VALUE"], errors="coerce")
    # df2.head()

    # wide format: one row per (ID, DATE)
    wide = df2.pivot_table(
        index=["ID", "DATE"], columns="ELEMENT", values="VALUE", aggfunc="first"
    ).reset_index()
    # wide.head()

    # merge metadata: QFLAG, SFLAG, OBSTIME
    meta_cols = df2[["ID", "DATE", "ELEMENT", "QFLAG", "SFLAG", "OBSTIME"]].copy()
    meta = meta_cols.pivot_table(
        index=["ID", "DATE"],
        columns="ELEMENT",
        values=["QFLAG", "SFLAG", "OBSTIME"],
        aggfunc="first",
    )
    meta.columns = [f"{a}_{b}" for a, b in meta.columns]
    meta = meta.reset_index()
    # combine numeric temperatures + metadata
    out = wide.merge(meta, on=["ID", "DATE"], how="left")
    # convert tenths degC to degC
    if "TMAX" in out:
        out["TMAX"] = out["TMAX"] / 10.0
    if "TMIN" in out:
        out["TMIN"] = out["TMIN"] / 10.0
    # out.head()

    return out


# %%
def load_station_meta(path=fpath / "ghcnd-stations.txt"):
    colspecs = [
        (0, 11),  # ID
        (12, 20),  # LAT
        (21, 30),  # LON
        (31, 37),  # ELEV
        (38, 40),  # STATE
        (41, 71),  # NAME
        (72, 75),  # GSN FLAG
        (76, 79),  # HCN/CRN FLAG
        (80, 85),  # WMO ID
    ]
    names = ["ID", "LAT", "LON", "ELEV", "STATE", "NAME", "GSN", "HCN", "WMO"]
    dfm = pd.read_fwf(path, colspecs=colspecs, names=names)
    dfm["ELEV"] = pd.to_numeric(dfm["ELEV"], errors="coerce")
    return dfm.set_index("ID")[["LAT", "LON", "ELEV"]]


# %%
def process_station(df_station, meta, year, month):
    """
    df_station: rows from df_clean for one station
    must contain columns:
    ID, DATE,
    TMAX, TMIN,
    OBSTIME_TMAX, OBSTIME_TMIN,
    SFLAG_TMAX, SFLAG_TMIN,

    returns dataframe with DATE, ID, LAT, LON, ELEV, TMAX_UTC, TMIN_UTC
    """
    sid = df_station["ID"].iloc[0]
    # --- station metadata --- #
    lat = meta.loc[sid, "LAT"]
    lon = meta.loc[sid, "LON"]
    elev = meta.loc[sid, "ELEV"]

    # --- sort by date --- #
    df = df_station.sort_values("DATE").reset_index(drop=True)

    # ========================================================
    # Step 1 - determine local reporting time
    # ========================================================
    def get_obstime_row(row, var):
        """
        Return integer HHMM or default 2400 or None
        """
        sflag = row[f"SFLAG_{var}"]
        obst = row[f"OBSTIME_{var}"]
        if sflag == "S":  # case-sensitive: Global Summary of the Day
            return None
        if pd.isna(obst) or str(obst).strip() == "":
            return int(2400)
        return int(obst)

    df["OBSTIME_local"] = [
        get_obstime_row(df.iloc[i], "TMAX")
        for i in range(len(df))  # use TMAX's SFLAG/OBSTIME convention
    ]

    # ========================================================
    # Step 2 - local time -> local solar time: TBD
    # currently: assume the local report time is on local solar time
    # ========================================================
    def solar_offset_minutes(lon_deg):
        return 4 * lon_deg  # minutes

    TMAX_local = df["TMAX"].to_numpy()
    TMIN_local = df["TMIN"].to_numpy()
    n = len(df)
    TMAX_LST = np.full(n, np.nan)
    TMIN_LST = np.full(n, np.nan)

    for i in range(n):
        obst = df["OBSTIME_local"].iloc[i]
        if obst is None:
            # SFLAG = "S"
            TMAX_LST[i] = TMAX_local[i]
            TMIN_LST[i] = TMIN_local[i]
            continue
        # Between 0600-1400 LST -> use previous day's TMAX
        if 600 <= obst < 1400:
            if i > 0:
                TMAX_LST[i] = TMAX_local[i - 1]
            TMIN_LST[i] = TMIN_local[i]
        else:
            TMAX_LST[i] = TMAX_local[i]
            TMIN_LST[i] = TMIN_local[i]

    # ========================================================
    # Step 3 - Local solar time -> UTC time adjustment
    # ========================================================
    delta_utc = solar_offset_minutes(lon) / 60.0  # convert minutes to hours
    TMAX_UTC = np.full(n, np.nan)
    TMIN_UTC = np.full(n, np.nan)
    for i in range(n):
        if 5 <= delta_utc <= 12:
            TMAX_UTC[i] = TMAX_LST[i]
            if i < (n - 1):
                TMIN_UTC[i] = TMIN_LST[i + 1]
        elif -12 <= delta_utc <= -10:
            if i > 0:
                TMAX_UTC[i] = TMAX_LST[i - 1]
            TMIN_UTC[i] = TMIN_LST[i]
        else:
            TMAX_UTC[i] = TMAX_LST[i]
            TMIN_UTC[i] = TMIN_LST[i]

    # filter out this month only
    yyyy = int(year)
    mm = int(month)
    start_date = yyyy * 10000 + mm * 100 + 1
    end_date = yyyy * 10000 + mm * 100 + 31
    df["DATE"] = df["DATE"].astype(int)
    mask_month = (df["DATE"] >= start_date) & (df["DATE"] <= end_date)
    df_month = df.loc[mask_month].copy()
    TMAX_UTC_month = TMAX_UTC[mask_month]
    TMIN_UTC_month = TMIN_UTC[mask_month]

    df_out = pd.DataFrame(
        {
            "ID": sid,
            "DATE": df_month["DATE"],
            "LAT": lat,
            "LON": lon,
            "ELEV": elev,
            "TMAX_UTC": TMAX_UTC_month,
            "TMIN_UTC": TMIN_UTC_month,
        }
    )
    return df_out


# %%
# load lat, lon, elev info for stations
meta = load_station_meta(fpath / "ghcnd-stations.txt")
# meta.head()

# loop over calendar months
for year in range(1980+start_idx, 1980+end_idx):
    for month in range(1, 13):
        # example of processing Jan 1980
        df_raw = extract_month_window(year, month)
        df_clean = extract_TMAX_TMIN_records(df_raw)
        # df_clean.head()
        del df_raw
        # apply to the entire dataset
        df_UTC_all = []
        for sid, df_st in df_clean.groupby("ID"):
            if sid not in meta.index:
                print(f"Did not find metadata for station {sid}")
                continue
            out = process_station(df_st, meta, year=year, month=month)
            df_UTC_all.append(out)

        df_UTC_final = pd.concat(df_UTC_all, ignore_index=True)
        # save output for each month
        if df_UTC_final is None or df_UTC_final.empty:
            print(f"Did not save data for {year}-{month:02d}")
            continue
        outfname = dir_output / f"UTC_{year}{month:02d}.parquet"
        df_UTC_final.to_parquet(outfname, index=True)
        ## sanity check to load back
        # df_try = pd.read_parquet(outfname)
