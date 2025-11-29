from kfp import dsl
from pipelines.components.train_arima_default_component_v2 import train_arima_default_component_v2


@dsl.pipeline(
    name="train-arima-default-pipeline-v2",
    description="Train ARIMA model in BigQuery (KFP v2)",
)
def train_arima_default_pipeline_v2(
    project_id: str = "ml-ai-portfolio",
    dataset_id: str = "taxi_forecasting",
    source_table: str = "train_2022",
    model_name: str = "daily_arima_default_model_v1",
):
    _ = train_arima_default_component_v2(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=source_table,
        model_name=model_name,
    )

