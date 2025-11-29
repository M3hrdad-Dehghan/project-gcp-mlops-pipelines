from kfp import dsl
from pipelines.components.evaluation_component_v2 import evaluation_component_v2


@dsl.pipeline(
    name="evaluate-arima-model-aic-pipeline-v2",
    description="Evaluate a BigQuery ARIMA model using AIC (KFP v2)",
)
def evaluation_pipeline_v2(
    project_id: str = "ml-ai-portfolio",
    model_path: str = "ml-ai-portfolio.taxi_forecasting.daily_arima_default_model_v1",
):
    _ = evaluation_component_v2(
        project_id=project_id,
        model_path=model_path,
    )

