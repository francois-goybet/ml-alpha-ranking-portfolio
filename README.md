# ml-alpha-ranking-portfolio

This project builds a leakage-aware stock ranking pipeline for monthly alpha prediction.

The workflow is split into two parts:
1. Run the notebook to query, clean, and assemble the dataset.
2. Run the Python pipeline to train the ranking model and generate plots and ranking diagnostics.

## Setup

### Create the Virtual Environment

To create a virtual environment named `ml-alpha-ranking-portfolio`, run:

```powershell
python -m venv ml-alpha-ranking-portfolio
```

### Activate the Virtual Environment

On Windows (PowerShell):

```powershell
.\ml-alpha-ranking-portfolio\Scripts\Activate.ps1
```

### Install Requirements

Once the virtual environment is activated, install the required packages:

```powershell
pip install -r requirements.txt
```

Alternatively, you can install the packages individually:

```powershell
pip install numpy pandas scikit-learn plotly matplotlib seaborn xgboost
```

## Workflow

### 1. Build the dataset from the notebook

Run [notebook/notebook1.ipynb](notebook/notebook1.ipynb).

The notebook is responsible for:
- querying the raw market and fundamentals data,
- merging them into a point-in-time dataset,
- producing the CSV used by the training pipeline.

In practice, it is enough to run `notebook1.ipynb` to generate the data needed by the project.

### 2. Run the training pipeline

Once the dataset has been created by the notebook, run:

```powershell
python .\main.py --config config/config.yaml
```

This script will:
- load the YAML configuration,
- split the dataset into train/validation/test periods,
- generate data exploration plots,
- train the XGBoost pairwise ranking model,
- export model diagnostics,
- export monthly ranking tables for the test period.

## Outputs

The pipeline saves its outputs under [generated/plots](generated/plots):
- data description plots,
- model training plots,
- feature importance plots,
- HTML tables and CSV files for monthly test rankings.