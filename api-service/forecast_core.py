# forecast_core.py

from typing import List, Dict, Optional
from google.cloud import bigquery
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import datetime


class BigQueryClient:
    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = project_id

        if self.project_id:
            self.client = bigquery.Client(project=self.project_id)
        else:
            self.client = bigquery.Client()

    def run_query(self, query: str):
        logging.info("Running query: %s", query)
        job = self.client.query(query)
        result = job.result()
        df = result.to_dataframe()
        logging.info("Query returned %s rows", len(df))
        return df


class BQMLTrainer:
    """Lightweight trainer used by the API to retrain a BQML ARIMA_PLUS model directly."""
    def __init__(self, project_id: str, dataset_id: str) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)

    def _full_model_path(self, model_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{model_name}"

    def _table_path(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    def get_max_date(self, source_table: str, date_col: str = 'trip_date') -> Optional[str]:
        table_path = self._table_path(source_table)
        q = f"SELECT MAX({date_col}) AS max_date FROM `{table_path}`"
        job = self.client.query(q)
        rs = job.result()
        for row in rs:
            if row['max_date'] is None:
                return None
            return str(row['max_date'].date()) if hasattr(row['max_date'], 'date') else str(row['max_date'])

    def train_arima(self, source_table: str, model_name: str, time_col: str = 'trip_date', value_col: str = 'total_trips', horizon: int = 30) -> str:
        full_model_path = self._full_model_path(model_name)
        full_table_path = self._table_path(source_table)
        logging.info("Called train_arima for model %s from %s with horizon=%s", model_name, source_table, horizon)
        query = f"""
            CREATE OR REPLACE MODEL `{full_model_path}`
            OPTIONS(
                model_type = 'ARIMA_PLUS',
                auto_arima = TRUE,
                time_series_timestamp_col = '{time_col}',
                time_series_data_col = '{value_col}',
                horizon = {horizon}
            ) AS
            SELECT
                {time_col},
                {value_col}
            FROM `{full_table_path}`
            ORDER BY {time_col};
        """
        job = self.client.query(query)
        job.result()
        return full_model_path


class ForecastCore:
    def __init__(self, model_full_path: str, project_id: Optional[str] = None) -> None:
        self.model_full_path = model_full_path
        self.bq = BigQueryClient(project_id=project_id)
        # Capture stats about the last query executed (for UI/helpful responses)
        self.last_query_stats = {
            "original_count": 0,
            "filtered_count": 0,
            "min_timestamp": None,
            "max_timestamp": None,
        }

    def _validate_inputs(self, start_date: str, horizon: int) -> datetime.date:
        # horizon: enforce 1..30 (November has 30 days in demo)
        if not isinstance(horizon, int) or horizon < 1 or horizon > 30:
            raise ValueError("horizon must be an integer between 1 and 30.")

        # start_date format and ensure it's within November 2022
        try:
            parsed_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        except Exception:
            raise ValueError("start_date must be in YYYY-MM-DD format.")

        min_date = datetime.date(2022, 11, 1)
        max_date = datetime.date(2022, 11, 30)
        if parsed_date < min_date or parsed_date > max_date:
            raise ValueError("start_date must be within November 2022 (2022-11-01 to 2022-11-30)")

        return parsed_date

    def forecast(self, start_date: str, horizon: int) -> List[Dict]:
        parsed_date = self._validate_inputs(start_date, horizon)

        def build_query(h):
            return f"""
        SELECT
            forecast_timestamp,
            forecast_value
        FROM
            ML.FORECAST(
                MODEL `{self.model_full_path}`,
                STRUCT({h} AS horizon)
            )
        """

        # Build the forecast for the requested horizon — no mapping, no out-of-range logic for the demo
        df = self.bq.run_query(build_query(horizon))

        self.last_query_stats["original_count"] = len(df)
        if df.empty:
            logging.info("Forecast query returned no rows for model %s with horizon %s", self.model_full_path, horizon)
            self.last_query_stats.update({"filtered_count": 0, "min_timestamp": None, "max_timestamp": None})
            return []

        # normalize
        df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"]).dt.tz_convert(None) if df["forecast_timestamp"].dt.tz is not None else pd.to_datetime(df["forecast_timestamp"])  # noqa
        min_ts = df["forecast_timestamp"].min()
        max_ts = df["forecast_timestamp"].max()
        logging.info("Forecast timestamp range: %s - %s", min_ts, max_ts)
        self.last_query_stats["min_timestamp"] = str(min_ts)
        self.last_query_stats["max_timestamp"] = str(max_ts)

        parsed_dt = pd.to_datetime(parsed_date)
        logging.info("Parsed start_date: %s (parsed_dt=%s)", parsed_date, parsed_dt)
        # Always filter by start_date for this demo
        df = df[df["forecast_timestamp"] >= parsed_dt]
        logging.info("After filtering by start_date (%s), %s rows remain", parsed_date, len(df))
        self.last_query_stats["filtered_count"] = len(df)

        # compute df window (no mapping)
        df_window = df.sort_values(by=["forecast_timestamp"]).reset_index(drop=True)

        if df_window.empty:
            logging.info("No rows after windowing/filters — nothing to return")
            return []

        # prepare output
        df_window["forecast_date"] = df_window["forecast_timestamp"].dt.date
        results = []
        for _, row in df_window.iterrows():
            results.append({"date": row["forecast_date"].isoformat(), "forecast": float(row["forecast_value"])})

        return results
