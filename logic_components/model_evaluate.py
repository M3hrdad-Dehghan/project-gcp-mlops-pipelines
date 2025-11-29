from google.cloud import bigquery


class ModelEvaluator:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def evaluate(self, model_path: str) -> dict:
        query = f"""
        SELECT aic
        FROM ML.ARIMA_EVALUATE(MODEL `{model_path}`)
        LIMIT 1
        """

        df = (
            self.client.query(query)
            .result()
            .to_dataframe()
        )

        if df.empty:
            raise RuntimeError(f"No evaluation results found for model: {model_path}")

        aic_value = float(df["aic"].iloc[0])

        return {
            "model": model_path,
            "aic": aic_value
        }
