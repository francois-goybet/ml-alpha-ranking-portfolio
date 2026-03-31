# ml-alpha-ranking-portfolio

This project aims to build an alpha-ranking pipeline for equities.

At this stage, the data layer is already structured and configurable. The model and full training/evaluation pipeline are the next main focus.

## Requirements

Install dependencies from [requirements.txt](requirements.txt):

```powershell
pip install -r requirements.txt
```

## Virtual Environment

Create a local virtual environment:

```powershell
python -m venv ml-alpha-ranking-portfolio
```

Activate it on Windows PowerShell:

```powershell
.\ml-alpha-ranking-portfolio\Scripts\Activate.ps1
```

Then install the requirements:

```powershell
pip install -r requirements.txt
```

## Current Data Architecture

The data module is organized around three classes:

1. `WRDSDataLoader` in [src/data/WRDSDataLoader.py](src/data/WRDSDataLoader.py)
- Responsibility: retrieve and merge raw datasets from WRDS.
- Contains source-specific methods (example stubs: price data, fundamentals data).

2. `FeatureBuilder` in [src/data/feature_builder.py](src/data/feature_builder.py)
- Responsibility: transform merged raw data into feature-enriched data.
- Includes one toy example function: `rolling_price`.

3. `DataManager` in [src/data/DataManager.py](src/data/DataManager.py)
- Responsibility: orchestrate data loading and feature construction.
- Instantiates both `WRDSDataLoader` and `FeatureBuilder`.
- Exposes a single `build_dataset(source=...)` entry point.

## Configuration

Main configuration lives in [config/config.yaml](config/config.yaml) with three sections:

1. `data`
- Data-level settings (for example CSV path and date ranges).

2. `model`
- Model hyperparameters and training options.

3. `pipeline`
- Pipeline-level behavior.
- `rebuild_dataset: true` means rebuild from WRDS.
- `rebuild_dataset: false` means read the existing CSV.

Config parsing is handled in [src/config/config_loader.py](src/config/config_loader.py), and consumed by [main.py](main.py).

## Run

Run with:

```powershell
python .\main.py --config config/config.yaml
```

## Next Priorities

Now that the data structure is in place, attention should move to:

1. Model training flow (fit/predict/evaluation integration).
2. Pipeline outputs and reporting.
3. End-to-end robustness (error handling, reproducibility, and tests).