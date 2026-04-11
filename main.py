import argparse

from src.config.config_loader import load_config

from src.data.DataManager import DataManager
from src.model.model import MultiHorizonRanker

def main(args):

    # Load configuration with lightweight parsing.
    config = load_config(args.config)
    data_manager = DataManager(config.get("data", {}))
    data_manager.get_data(start=config["data"].get("train_start", "1990-01-01"), end=config["data"].get("test_end", "2024-12-31"), market_cap=config["data"].get("market_cap", 10))
    s = data_manager.get_train_val_test(targets=["ret_1m", "ret_3m", "ret_6m"])
    # print stats about the splits
    for split_name, (X, y, group) in s.items():

        print(f"{split_name}: {len(X)} samples, {len(X.columns)} features, {len(set(group))} groups.")
        # print columns of y
        print(f"  Targets: {y.columns.tolist()}")
        # print number of nan values in y
        print(f"  NaN values in targets: {y.isna().sum().to_dict()}")
        # print firsts columns of X
        print(f"  First 5 columns of X: {X.columns[:5].tolist()}")
        # print group first 5 values
        print(f"  First 5 group values: {group[:5]}")
        # print size of the first group
        print(f"  Size of first group: {group[0]}")
        print()
    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

    # Check if some values in y_train are NaN (should not happen with current data loading, but just in case).
    if y_train.isna().any().any():
        print("Warning: NaN values found in y_train. This may cause issues during training.")
        print(y_train.isna().sum())

    model = MultiHorizonRanker(targets=["ret_1m"], **config.get("model", {}))
    model.fit(X_train, y_train, group_train, (X_val, y_val), group_val, verbose=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate AlphaXGBoost model for stock ranking.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_francois.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    main(args)
