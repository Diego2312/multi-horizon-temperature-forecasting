# Rome Ciampino Daily Temperature Data

## Overview
This dataset contains daily meteorological observations from the **Rome Ciampino** station, provided by the NOAA Global Historical Climatology Network – Daily (GHCN-Daily).  
The **TAVG** (daily mean temperature) variable in degrees Celsius is used as the prediction target.

- **Station ID:** IT000016239  
- **Coordinates:** 41.799° N, 12.594° E  
- **Temporal coverage (full dataset):** 1951–present  
- **Source:** [NOAA NCEI – GHCN Daily Summaries](https://www.ncei.noaa.gov/data/daily-summaries/)

## Download Instructions

### Automated Download (Recommended)
Run the following command from the project root:
```bash
python -m data.download_data
```
### Manual Download
1. Open the dataset link:
https://www.ncei.noaa.gov/data/daily-summaries/access/IT000016239.csv

2. Save the file as 'Raw_dataset.csv' in the data/ directory.

## Dataset info
https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00861/html



