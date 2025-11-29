from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery"],
)
def train_arima_default_component_v2(
    project_id: str,
    dataset_id: str,
    source_table: str,
    model_name: str,
) -> str:
    """
    Train ARIMA_PLUS model in BigQuery using the BQML trainer, return full model path.
    """
    # local imports: try importing trainer from logic_components; otherwise provide a fallback
    try:
        from logic_components.model_trainer import BQMLTrainer
    except Exception:
        from google.cloud import bigquery

        class BQMLTrainer:
            def __init__(self, project_id: str, dataset_id: str) -> None:
                self.project_id = project_id
                self.dataset_id = dataset_id
                self.client = bigquery.Client(project=self.project_id)

            def _full_model_path(self, model_name: str) -> str:
                return f"{self.project_id}.{self.dataset_id}.{model_name}"

            def _table_path(self, table_name: str) -> str:
                return f"{self.project_id}.{self.dataset_id}.{table_name}"

            def train_arima(self, source_table: str, model_name: str, time_col: str = "trip_date", value_col: str = "total_trips", horizon: int = 30) -> str:
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
                return full_model_path

    trainer = BQMLTrainer(project_id=project_id, dataset_id=dataset_id)

    model_path = trainer.train_arima(
        source_table=source_table,
        model_name=model_name,
    )

    print(f"Trained model: {model_path}")
    return model_path

