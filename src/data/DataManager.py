from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import wrds

_DATA_DIR = Path("data")
_CONSTRUCTION_DIR = _DATA_DIR / "construction"
_DATASET_PARQUET = _DATA_DIR / "dataset.parquet"
_SIGNAL_DOC_PARQUET = _DATA_DIR / "signal_doc.parquet"
_RF_PARQUET = _DATA_DIR / "rf.parquet"
_RET_SP500_PARQUET = _DATA_DIR / "ret_sp500.parquet"
_CHUNK_SIZE = 20
_META = {"permno", "yyyymm", "ret", "market_cap_musd", "sector", "ret_1m", "ret_3m", "ret_6m"}


class DataManager:
    """Builds the ML dataset from WRDS (CRSP) and OpenAssetPricing signals."""

    def __init__(self, data_config: dict[str, Any]) -> None:
        self.data_config = data_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signal_doc(self) -> pd.DataFrame:
        """Return the OpenAssetPricing signal documentation.

        If ``data/signal_doc.parquet`` already exists it is loaded from disk.
        Otherwise the doc is downloaded, saved to ``data/signal_doc.parquet``,
        and returned.

        Returns
        -------
        pd.DataFrame
            Signal documentation table (one row per signal, includes
            ``Acronym``, ``SignalName``, ``CategoryDescription``, …).
        """
        if _SIGNAL_DOC_PARQUET.exists():
            return pd.read_parquet(_SIGNAL_DOC_PARQUET)

        import openassetpricing as oap

        openap = oap.OpenAP()
        signal_doc = openap.dl_signal_doc("pandas")

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        signal_doc.to_parquet(_SIGNAL_DOC_PARQUET, index=False)

        return signal_doc

    def get_data(
        self,
        start: str,
        end: str,
        market_cap: float = 100.0,
    ) -> pd.DataFrame:
        """Build (or reload) the full ML dataset.

        Parameters
        ----------
        start:
            Start date for the CRSP query, e.g. ``'1990-01-01'``.
        end:
            End date for the CRSP query.  Must not exceed ``'2024-12-31'``.
        market_cap:
            Minimum market capitalisation filter in millions USD (default 100).

        Returns
        -------
        pd.DataFrame
            The final merged dataset.  If ``data/dataset.parquet`` already exists
            it is returned immediately and all intermediate construction files
            are removed.
        """
        end_year = int(str(end)[:4])
        if end_year > 2024:
            raise ValueError(f"end date must not exceed 2024, got {end!r}.")

        # If the final dataset already exists, clean up and return it.
        if _DATASET_PARQUET.exists():
            self._clean_construction_files()
            
            return pd.read_parquet(_DATASET_PARQUET)

        _CONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Pull CRSP monthly returns from WRDS.
        base = self._load_crsp(start, end, market_cap)

        # 2. Load OpenAP signals in chunks and save intermediate parquets.
        self._build_openap_chunks(base)

        # 3. Merge all parquet chunks into the final dataset.
        final_df = self._merge_construction_parquets()

        # 4. Persist and clean up.
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(_DATASET_PARQUET, index=False)
        self._clean_construction_files()

        return final_df

    def get_split(
        self,
        split: str,
        targets: list[str] = ["ret_1m", "ret_3m", "ret_6m"],
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        """Return (X, y, groups) for a single split with preprocessing.

        Note: when using this method standalone, preprocessing is fitted and
        applied on the same split (no look-ahead issue for a single split).
        For proper train-fit / val-test-apply, use :meth:`get_train_val_test`.

        Parameters
        ----------
        split:
            One of ``"train"``, ``"val"``, ``"test"``.
        targets:
            List of target column names, e.g. ``["ret_1m", "ret_3m", "ret_6m"]``.
        start:
            Override start date (``YYYY-MM-DD``).  If None, uses config value.
        end:
            Override end date (``YYYY-MM-DD``).  If None, uses config value.

        Returns
        -------
        X : pd.DataFrame
            Preprocessed feature matrix, sorted by yyyymm then permno.
        y : pd.DataFrame
            Target DataFrame aligned with X.
        groups : list[int]
            Number of stocks per month; ``sum(groups) == len(X)``.
        """
        X, y, groups = self._get_raw_split(split, targets, start=start, end=end)
        X = self._preprocess_features(X, fit=True)
        return X, y, groups

    def get_train_val_test(
        self,
        targets: list[str] = ["ret_1m", "ret_3m", "ret_6m"],
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[int]]]:
        """Return train, val and test splits with preprocessing fit on train only.

        Fits any transformation (scaling, imputation…) on the training set and
        applies the same fitted transform to val and test — avoiding look-ahead
        bias.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"``, each containing
        ``(X, y, groups)``.
        """
        X_train, y_train, groups_train = self._get_raw_split("train", targets)
        X_val,   y_val,   groups_val   = self._get_raw_split("val",   targets)
        X_test,  y_test,  groups_test  = self._get_raw_split("test",  targets)

        # Fit preprocessing on train, apply to all splits
        X_train = self._preprocess_features(X_train, fit=True)
        X_val   = self._preprocess_features(X_val,   fit=False)
        X_test  = self._preprocess_features(X_test,  fit=False)

        return {
            "train": (X_train, y_train, groups_train),
            "val":   (X_val,   y_val,   groups_val),
            "test":  (X_test,  y_test,  groups_test),
        }

    def get_rf(self, start):
        
        if _RF_PARQUET.exists():
            return pd.read_parquet(_RF_PARQUET)

        db = wrds.Connection()

        query = f"""
        select
            extract(year from date)*100 + extract(month from date) as yyyymm,
            rf
        from ff.factors_monthly
        where date >= '{start}'
        order by date
        """

        df = db.raw_sql(query)

        # convert rf from % to decimal
        df["rf"] = df["rf"] / 100.0
        df.to_parquet(_RF_PARQUET, index=False)
        return df

    def get_ret_sp500(self, start):

        if _RET_SP500_PARQUET.exists():
            return pd.read_parquet(_RET_SP500_PARQUET)
        
        db = wrds.Connection()

        df = db.raw_sql(f"""
            SELECT date, sprtrn
            FROM crsp.msi
            where date >= '{start}'
        """)

        df['yyyymm'] = pd.to_datetime(df['date']).dt.strftime('%Y%m')

        df = df[['yyyymm', 'sprtrn']].rename(columns={'sprtrn': 'ret'})
        df.to_parquet(_RET_SP500_PARQUET, index=False)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_raw_split(
        self,
        split: str,
        targets: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        """Internal version of get_split — returns raw (unpreprocessed) X."""
        if _DATASET_PARQUET.exists():
            df = pd.read_parquet(_DATASET_PARQUET)
        else:
            raise FileNotFoundError(
                "dataset.parquet not found. Call get_data() first."
            )

        # Add lagged monthly return features computed on the full dataset
        # (before date filtering) so lags at the start of each split are correct.
        return_lags = self.data_config.get("return_lags", list(range(1, 7)))
        df = df.sort_values(["permno", "yyyymm"])
        for lag in return_lags:
            df[f"ret_lag{lag}"] = df.groupby("permno")["ret"].shift(lag)

        start_date = start or self.data_config.get(f"{split}_start")
        end_date = end or self.data_config.get(f"{split}_end")

        if start_date is None or end_date is None:
            raise ValueError(
                f"No dates for split '{split}'. Pass start/end or set "
                f"'{split}_start' / '{split}_end' in config."
            )

        start_ym = int(str(start_date).replace("-", "")[:6])
        end_ym = int(str(end_date).replace("-", "")[:6])

        df = (
            df.loc[(df["yyyymm"] >= start_ym) & (df["yyyymm"] <= end_ym)]
            .sort_values(["yyyymm", "permno"])
            .reset_index(drop=True)
        )

        feature_cols = [c for c in df.columns if c not in _META]

        targets_list = [targets] if isinstance(targets, str) else list(targets)
        df = df.dropna(subset=targets_list).reset_index(drop=True)
        X = df.copy()
        y = df[targets]
        groups = df.groupby("yyyymm").size().tolist()
        return X, y, groups

    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Clean the feature matrix before passing it to the ranker.

        When ``fit=True`` (train set), compute and store any statistics needed
        for the transform. When ``fit=False`` (val/test), apply the previously
        fitted transform without recomputing anything.

        Steps applied:
          - replace ±inf with NaN
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        # Convert any pandas nullable / extension dtypes to plain numpy float64
        # so XGBoost/LightGBM can consume the array without TypeError on pd.NA.
        X = X.apply(lambda col: col.astype("float64") if ((not col.dtype == np.float64) and (col.name not in _META)) else col)
        return X

    def _load_crsp(self, start: str, end: str, market_cap: float) -> pd.DataFrame:
        """Query CRSP monthly stock file from WRDS and compute lagged returns.

        Steps
        -----
        1. Connect to WRDS.
        2. Pull ``crsp.msf`` filtered by market cap and date range.
        3. Keep monthly market capitalisation in millions USD.
        4. Build a ``yyyymm`` integer key.
        5. Compute forward returns: ``ret_1m``, ``ret_3m``, ``ret_6m``.
        """
        import wrds

        db = wrds.Connection(wrds_username=self.data_config.get("wrds_username", ""))
        query = f"""
            SELECT
                permno,
                date,
                ret,
                prc,
                shrout,
                abs(prc * shrout) / 1e6 AS market_cap_musd
            FROM crsp.msf
            WHERE abs(prc * shrout) / 1e6 > {market_cap}
              AND date >= '{start}'
              AND date <= '{end}'
            ORDER BY permno, date
        """
        df = db.raw_sql(query)
        db.close()

        df = self._prepare_crsp(df)
        return df

    @staticmethod
    def _prepare_crsp(df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw CRSP output and add forward-return columns.

        Parameters
        ----------
        df:
            Raw dataframe returned by the WRDS CRSP query (must contain
            ``permno``, ``date``, ``ret`` columns).

        Returns
        -------
        pd.DataFrame
            Columns: ``permno``, ``yyyymm``, ``ret``, ``market_cap_musd``,
            ``ret_1m``, ``ret_3m``, ``ret_6m``.
        """
        df["date"] = pd.to_datetime(df["date"])
        df["yyyymm"] = df["date"].dt.strftime("%Y%m").astype(int)
        df["permno"] = df["permno"].astype(int)

        df = df[["permno", "yyyymm", "ret", "market_cap_musd"]].sort_values(
            ["permno", "yyyymm"]
        )

        df["ret_1m"] = df.groupby("permno")["ret"].shift(-1)
        df["ret_3m"] = df.groupby("permno")["ret"].shift(-3)
        df["ret_6m"] = df.groupby("permno")["ret"].shift(-6)

        return df

    def _build_openap_chunks(self, base: pd.DataFrame) -> None:
        """Download OpenAP signals in batches and save one parquet per batch.

        Each parquet at ``data/construction/ml_dataset_part_XXXX.parquet``
        contains the CRSP base columns plus the signals of that batch,
        left-merged on ``(permno, yyyymm)``.

        Parameters
        ----------
        base:
            CRSP base dataframe returned by :meth:`_load_crsp`.
        """
        import openassetpricing as oap
        from tqdm import tqdm

        openap = oap.OpenAP()
        signal_doc = self.get_signal_doc()
        signals = signal_doc.Acronym.tolist()

        for i in tqdm(range(0, len(signals), _CHUNK_SIZE), desc="OpenAP batching"):
            
            print(f"Processing signals {i} to {min(i + _CHUNK_SIZE, len(signals))}...")
            batch = signals[i : i + _CHUNK_SIZE]
            feat = openap.dl_signal("pandas", batch)

            feat["permno"] = feat["permno"].astype(int)
            feat["yyyymm"] = feat["yyyymm"].astype(int)

            chunk = base.merge(feat, on=["permno", "yyyymm"], how="left")

            out_path = _CONSTRUCTION_DIR / f"ml_dataset_part_{i:04d}.parquet"
            chunk.to_parquet(out_path, index=False)

            del feat, chunk

    @staticmethod
    def _merge_construction_parquets() -> pd.DataFrame:
        """Read all chunk parquets and outer-merge them on common base columns.

        Returns
        -------
        pd.DataFrame
            Full dataset deduplicated on ``(permno, yyyymm)``.
        """
        files = sorted(_CONSTRUCTION_DIR.glob("ml_dataset_part_*.parquet"))
        if not files:
            raise RuntimeError("No construction parquet files found to merge.")

        base_cols = [
            "permno",
            "yyyymm",
            "ret",
            "market_cap_musd",
            "ret_1m",
            "ret_3m",
            "ret_6m",
        ]
        dfs = [pd.read_parquet(f) for f in files]

        merged = reduce(
            lambda left, right: pd.merge(left, right, on=base_cols, how="outer"),
            dfs,
        )
        return merged.drop_duplicates(subset=["permno", "yyyymm"])

    @staticmethod
    def _clean_construction_files() -> None:
        """Delete all parquet files in the construction directory.

        Also removes the directory itself if it is left empty.
        """
        if not _CONSTRUCTION_DIR.exists():
            return
        for f in _CONSTRUCTION_DIR.glob("ml_dataset_part_*.parquet"):
            f.unlink()
        try:
            _CONSTRUCTION_DIR.rmdir()
        except OSError:
            pass

    def _get_sector_mapping(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Get a mapping from (permno, date) to sector using WRDS CRSP-Compustat link.
        Steps:
        1. Extract unique permnos from the dataset.
        2. Connect to WRDS and query the CRSP-Compustat link table to get SIC codes.
        3. Convert SIC codes to sectors.
        4. Return a DataFrame with columns (permno, date, sector).
        """
        permnos = dataset["permno"].unique().tolist()  # ou ta liste
        permno_str = ",".join(map(str, permnos))

        dataset["date"] = pd.to_datetime(dataset["yyyymm"].astype(str) + "01")
        import wrds
        db = wrds.Connection()


        query = f"""
        SELECT
            l.lpermno AS permno,
            l.linkdt,
            l.linkenddt,
            c.sic
        FROM crsp.ccmxpf_linktable l
        JOIN comp.company c
            ON l.gvkey = c.gvkey
        WHERE l.lpermno IN ({permno_str})
        AND l.linktype IN ('LU','LC')
        AND l.linkprim IN ('P','C')
        """

        map_df = db.raw_sql(query)

        # =========================
        # DATE CONVERSION
        # =========================
        map_df["linkdt"] = pd.to_datetime(map_df["linkdt"])
        map_df["linkenddt"] = pd.to_datetime(map_df["linkenddt"])

        # =========================
        # MERGE (time-consistent join)
        # =========================
        df = dataset.merge(map_df, on="permno", how="left")

        df = df[
            (df["date"] >= df["linkdt"]) &
            (df["date"] <= df["linkenddt"].fillna(pd.Timestamp("2099-12-31")))
        ]

        # =========================
        # SIC → SECTOR
        # =========================
        def sic_to_sector(sic):
            if pd.isna(sic):
                return "Unknown"
            elif 1000 <= sic < 1500:
                return "Energy"
            elif 1500 <= sic < 3000:
                return "Industrials"
            elif 3000 <= sic < 4000:
                return "Manufacturing"
            elif 4000 <= sic < 5000:
                return "Utilities"
            elif 5000 <= sic < 6000:
                return "Retail"
            elif 6000 <= sic < 7000:
                return "Financials"
            elif 7000 <= sic < 8000:
                return "Tech"
            elif 8000 <= sic < 9000:
                return "Healthcare"
            else:
                return "Other"

        df["sector"] = df["sic"].astype(int).apply(sic_to_sector)

        # =========================
        # FINAL OUTPUT (optional mapping)
        # =========================
        df.drop(columns=["linkdt", "linkenddt", "sic", "date"], inplace=True)
        cols = df.columns.tolist()
        cols.insert(3, cols.pop(cols.index("sector")))
        df = df[cols]
        return df