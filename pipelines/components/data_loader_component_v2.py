from kfp import dsl
from typing import Tuple



@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery", "pandas", "db-dtypes"
    ],
)
def data_loader_component_v2(
    project_id: str,
    dataset_id: str,
    source_table: str,
    train_table: str,
    test_table: str,
    cutoff_date: str,
) -> str:

    # Try to import DataLoader from the project's logic_components package; if not present at runtime,
    # fall back to an inline minimal implementation so the component can run in Vertex AI's container.
    try:
        from logic_components.data_loader import DataLoader
    except Exception:
        # Local inline fallback DataLoader class definition (minimal implementation)
        import pandas as pd
        from google.cloud import bigquery

        class DataLoader:
            def __init__(self, project_id: str, dataset_id: str, table_id: str, date_column: str = "trip_date") -> None:
                self.project_id = project_id
                self.dataset_id = dataset_id
                self.table_id = table_id
                self.date_column = date_column

            def load_data(self) -> pd.DataFrame:
                client = bigquery.Client(project=self.project_id)
                table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                query = f"""
                    SELECT *
                    FROM `{table_ref}`
                    ORDER BY {self.date_column}
                """
                df = client.query(query).to_dataframe()
                return df

            def train_test_split(self, df: pd.DataFrame, train_end_date: str):
                df = df.copy()
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                train_cutoff = pd.to_datetime(train_end_date)
                train_df = df[df[self.date_column] <= train_cutoff].copy()
                test_df = df[df[self.date_column] > train_cutoff].copy()
                return train_df, test_df

            def save_to_bigquery(self, df: pd.DataFrame, target_table: str, write_disposition: str = "WRITE_TRUNCATE") -> None:
                client = bigquery.Client(project=self.project_id)
                table_id = f"{self.project_id}.{self.dataset_id}.{target_table}"
                job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
                load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
                load_job.result()

    loader = DataLoader(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=source_table,
    )

    df = loader.load_data()
    train_df, test_df = loader.train_test_split(df, cutoff_date)
    loader.save_to_bigquery(train_df, train_table)
    loader.save_to_bigquery(test_df, test_table)

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    return train_table

