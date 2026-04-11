import argparse
from turtle import pd

from src.config.config_loader import load_config

from src.data.DataManager import DataManager
from src.model.model import HorizonEnsemble, MultiHorizonRanker

def main(args):

    # Load configuration with lightweight parsing.
    config = load_config(args.config)
    data_manager = DataManager(config.get("data", {}))
    data_manager.get_data(start=config["data"].get("train_start", "1990-01-01"), end=config["data"].get("test_end", "2024-12-31"), market_cap=config["data"].get("market_cap", 10))
    s = data_manager.get_train_val_test(targets=["ret_1m", "ret_3m", "ret_6m"])
 
    X_train, y_train, group_train = s["train"]
    X_val, y_val, group_val = s["val"]
    X_test, y_test, group_test = s["test"]

    model = MultiHorizonRanker(targets=["ret_1m", "ret_3m", "ret_6m"], **config.get("model", {}))
    model.fit(X_train, y_train, group_train, (X_val, y_val), group_val, verbose=True)

    horizon_ensemble = HorizonEnsemble(multi_ranker=model, weights=[0.5, 0.3, 0.2])  # Example weights for the horizons
    y_rank = horizon_ensemble.predict(X_test, group_test)    
    # concat X_test,_ y_test, y_rank into a single dataframe and save to csv
    df_test = X_test.copy()
    df_test["ret_1m"] = y_test["ret_1m"]
    df_test["ret_3m"] = y_test["ret_3m"]
    df_test["ret_6m"] = y_test["ret_6m"]
    df_test["rank_score"] = y_rank
    df_test.to_csv("generated/test_predictions.csv", index=False)
    # same for train and val
    df_train = X_train.copy()
    df_train["ret_1m"] = y_train["ret_1m"]
    df_train["ret_3m"] = y_train["ret_3m"]
    df_train["ret_6m"] = y_train["ret_6m"]
    y_rank_train = horizon_ensemble.predict(X_train, group_train)
    df_train["rank_score"] = y_rank_train
    df_train.to_csv("generated/train_predictions.csv", index=False)
    df_val = X_val.copy()
    df_val["ret_1m"] = y_val["ret_1m"]
    df_val["ret_3m"] = y_val["ret_3m"]
    df_val["ret_6m"] = y_val["ret_6m"]
    y_rank_val = horizon_ensemble.predict(X_val, group_val)
    df_val["rank_score"] = y_rank_val
    df_val.to_csv("generated/val_predictions.csv", index=False)

    
    feature_importances = model.get_feature_importance(importance_type="gain")
    # Saving feature importances (a dict of target to importance array) to a csv file
    print(feature_importances)
    # Saving it
    for target, importance in feature_importances.items():
        df_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance
        })
        df_importance.to_csv(f"generated/feature_importance_{target}.csv", index=False)

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