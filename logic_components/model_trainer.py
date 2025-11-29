from typing import Optional
from google.cloud import bigquery


class BQMLTrainer:
    def __init__(self, project_id: str, dataset_id: str) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)

    # -----------------------------------------------------------
    # Utility: Full BigQuery model path
    # -----------------------------------------------------------
    def _full_model_path(self, model_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{model_name}"

    def _table_path(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    # -----------------------------------------------------------
    # Train default ARIMA_PLUS model
    # -----------------------------------------------------------
    def train_arima(
        self,
        source_table: str,
        model_name: str,
        time_col: str = "trip_date",
        value_col: str = "total_trips",
        horizon: int = 30,
    ) -> str:
        """
        Trains a default ARIMA_PLUS model using BigQuery ML.
        """

        full_model_path = self._full_model_path(model_name)
        full_table_path = self._table_path(source_table)

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

        print(f"[ModelTrainer] ARIMA_PLUS model trained on table: {source_table}")
        print(f"[ModelTrainer] Model created: {full_model_path}")

        return full_model_path
