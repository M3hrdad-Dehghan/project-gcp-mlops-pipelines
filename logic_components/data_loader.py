from typing import Tuple
import pandas as pd
from google.cloud import bigquery


class DataLoader:
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        date_column: str = "trip_date",
    ) -> None:

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.date_column = date_column

    # ----------------------------
    # Load data from BigQuery
    # ----------------------------
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

    # ----------------------------
    # Train / test split by date
    # ----------------------------
    def train_test_split(
        self,
        df: pd.DataFrame,
        train_end_date: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        train_cutoff = pd.to_datetime(train_end_date)

        train_df = df[df[self.date_column] <= train_cutoff].copy()
        test_df = df[df[self.date_column] > train_cutoff].copy()

        return train_df, test_df

    # ----------------------------
    # Save DataFrame to BigQuery
    # ----------------------------
    def save_to_bigquery(
        self,
        df: pd.DataFrame,
        target_table: str,
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> None:

        client = bigquery.Client(project=self.project_id)

        table_id = f"{self.project_id}.{self.dataset_id}.{target_table}"

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
        )

        load_job = client.load_table_from_dataframe(
            df,
            table_id,
            job_config=job_config,
        )
        load_job.result()

        print(f"[DataLoader] Saved {len(df)} rows to {table_id}")
