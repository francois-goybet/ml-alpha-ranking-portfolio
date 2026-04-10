from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_DIR = Path("data")
_CONSTRUCTION_DIR = _DATA_DIR / "construction"
_DATASET_PARQUET = _DATA_DIR / "dataset.parquet"
_SIGNAL_DOC_PARQUET = _DATA_DIR / "signal_doc.parquet"
_CHUNK_SIZE = 20


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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_crsp(self, start: str, end: str, market_cap: float) -> pd.DataFrame:
        """Query CRSP monthly stock file from WRDS and compute lagged returns.

        Steps
        -----
        1. Connect to WRDS.
        2. Pull ``crsp.msf`` filtered by market cap and date range.
        3. Build a ``yyyymm`` integer key.
        4. Compute forward returns: ``ret_1m``, ``ret_3m``, ``ret_6m``.
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
            Columns: ``permno``, ``yyyymm``, ``ret``, ``ret_1m``, ``ret_3m``,
            ``ret_6m``.
        """
        df["date"] = pd.to_datetime(df["date"])
        df["yyyymm"] = df["date"].dt.strftime("%Y%m").astype(int)
        df["permno"] = df["permno"].astype(int)

        df = df[["permno", "yyyymm", "ret"]].sort_values(["permno", "yyyymm"])

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

        base_cols = ["permno", "yyyymm", "ret", "ret_1m", "ret_3m", "ret_6m"]
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
