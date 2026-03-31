import argparse

from src.config.config_loader import load_config

from src.data.DataManager import DataManager
from src.model.model import AlphaXGBoost
from src.visualization import data_plots
from src.visualization import model_plots

def main(args):

    # Load configuration with lightweight parsing.
    config = load_config(args.config)
    data_manager = DataManager(config.get("data", {}))
    pipeline_cfg = config.get("pipeline", {})
    rebuild_dataset = bool(pipeline_cfg.get("rebuild_dataset", False))

    source = "wrds" if rebuild_dataset else "csv"
    dataset = data_manager.build_dataset(source=source)

    print(f"Dataset source: {source} | rows: {len(dataset)}")
    

    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate AlphaXGBoost model for stock ranking.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    main(args)