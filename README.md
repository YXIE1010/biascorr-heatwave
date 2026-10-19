# Bias correction for improved heatwave forecasts using NSF NCAR CREDIT

Code accompanying: [Yan Xie, Amy McGovern, Ann Bostrom, Douglas Rao], [Improving AI forecasts through bias correction: Application to heatwaves], [Artificial Intelligence for the Earth Systems], submitted 2026.

## Contents
- `scripts/` – process raw ERA5 / GHCNd / CREDIT output into intermediate files
- `notebooks/` – generate manuscript figures from intermediate files
Processing scripts were run on an HPC cluster via SLURM; job scripts are not included.

## Environment
- conda env create -f environment_processing.yml: to run notebooks and processing scripts  
- conda env create -f environment_credit.yml: to train AI weather models using NSF NCAR CREDIT platform

## Data and more
- ERA5 reanalysis: Google ARCO ERA5 cloud archive
- GHCNd station observations: NOAA NCEI public archive
- CREDIT model: NCAR MILES CREDIT, version/commit v2026.01
