# Training data shortlist

Sources checked **2026-11-03**. Dataset availability is not evidence that the
datasets have been downloaded, harmonized, or used to retrain Sentinel. Keep
source versions, checksums, attribution, coverage masks, and retrieval dates
with every prepared cache. The repository's MIT license does not relicense data.

| Source | Coverage / resolution | Role and access | Main constraint |
| --- | --- | --- | --- |
| [FPA-FOD 7th edition](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.7) | US, 1992–2024; 2.66 million reported fires | First choice for discovery labels. Public SQLite archive, approximately 220 MB compressed; government-funded data usable with attribution. | Discovery is a proxy for ignition; reporting and positional accuracy vary. Exclude final size, containment, and cause determinations from forecast inputs. |
| [Mesogeos](https://github.com/Orion-AI-Lab/mesogeos) | Mediterranean, 2006–2022; daily, 1 km | External danger-forecasting benchmark with weather, vegetation, human factors, and EFFIS labels. CC BY 4.0; current repository links to Google Drive. | Its burned-area and ignition-point values can encode eventual fire size. Derive a binary target, never feed those outcome values to the model. The former public S3 endpoint is discontinued. |
| [IberFire](https://github.com/JulenErcibengoa/IberFire), [data record](https://doi.org/10.5281/zenodo.15798999) | Spain, December 2007–December 2024; daily, 1 km | Regional transfer / newer-year stress test. Obtain the versioned datacube and its data terms from Zenodo. | Separate code and data licenses. Published fire labels omit small fires below the dataset's size criterion; they are not interchangeable with all-fire FPA-FOD labels. |
| [SeasFire Cube v0.3](https://zenodo.org/records/8055879), [project resources](https://seasfire.hua.gr/resources/) | Global, 2001–2021; 8-day, 0.25° grid | Seasonal pretraining and large-scale climate/vegetation context. Open versioned download; retain the record's license and citation. | Eight-day aggregates cannot establish exact seven-day ignition labels. Future aggregates, burned area, radiative power, and emissions must not enter pre-event features. |
| [NASA FIRMS VIIRS archives](https://firms.modaps.eosdis.nasa.gov/download/) | Global; 375 m; sensor-dependent record starts in 2012, 2018, or 2023 | Auxiliary active-fire detection labels and cross-sensor audits. Archive requests require Earthdata or email authentication. | Hotspots can be ongoing fires, prescribed burns, or other heat sources. Missing detections are not confirmed negatives. Prefer standard science-quality records for final benchmarks. |
| [MODIS MCD64A1 v6.1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD64A1) | Global; monthly, 500 m with estimated burn day | Burned-area label audit / separate burned-area task. NASA product, unrestricted subsequent use; Earth Engine requires access registration. | Apply valid-land, QA, date-uncertainty, and mapping-period masks. It uses MODIS active-fire observations, so agreement with MODIS hotspots is not independent ground truth. |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) | Global, 1950 onward; hourly, 0.1° | Add soil moisture, humidity, and drought-history features. CDS account/API and dataset terms; CC BY attribution. | Retrospective reanalysis is not the weather product available at forecast issue time. Preserve units and latency; evaluate archived operational inputs before deployment. |
| [CEMS fire-danger reanalysis](https://www.ecmwf.int/en/forecasts/datasets/fire-danger-indices-historical-data-copernicus-emergency-management-service) | Global, ERA5-driven historical indices | Physical benchmark: FWI and its moisture/drought components. Obtain through the Early Warning Data Store with its account and terms. | A modeled danger index is a predictor or benchmark, not observed ignition ground truth. Compare on the same grid, time horizon, and label mask. |
| [WildfireSpreadTS](https://papers.neurips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html) | US, 2018–2021; 607 events, daily, 375 m | Separate spread-forecasting experiment; weather, fuels, terrain, and active-fire time series. Follow the paper's dataset download and license. | Conditions on an already active fire. Do not add these samples to ignition training as if they represented unconditional risk. |

## Acquisition order

1. Start with the versioned FPA-FOD SQLite file and the existing Open-Meteo
   weather client. `sentinel prepare-climate` retains every cell in a fixed bbox,
   rather than selecting cells because they burned. It records the source file
   hash, citation, dates, grid, feature names, and target horizon in a local NPZ.
2. Benchmark on later years and unseen spatial blocks before increasing model
   size. Keep all non-event windows in calibration, alarm tuning, and testing.
3. Add humidity, soil moisture, terrain, fuels, and land cover as pre-issue-time
   features; freeze and version the changed feature schema. These adapters are
   future work, not part of the current seven-feature cache.
4. Evaluate Mediterranean/Spanish transfer separately. Match the target and
   source reporting thresholds first. Do not pool incompatible labels to inflate
   sample counts. IberFire and Mesogeos may share upstream European sources;
   deduplicate events before using either for training and the other for testing.
5. Use FIRMS / burned-area products for detection and label-quality experiments
   with explicit observation masks. Export a new, explicitly named task when
   evaluating spread or seasonal burned area.

For FPA-FOD 7, the published SQLite ZIP SHA-256 is
`a5d691bdbcf5a3de6a7e0d94380df311dd2c552645395312e55510424a7483fb`.
The archive expands to roughly 1 GB; use a regional subset when local storage is
limited, and record the original archive checksum as well as the subset checksum.

No-fire labels mean **no reported discovery in covered data**, not proof that
ignition was impossible. Treat missing source coverage as unknown, not zero.
