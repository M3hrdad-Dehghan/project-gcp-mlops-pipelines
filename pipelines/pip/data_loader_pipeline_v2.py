from kfp import dsl
from pipelines.components.data_loader_component_v2 import data_loader_component_v2


@dsl.pipeline(
    name="data-loader-pipeline-v2",
    description="Load & Split dataset using BigQuery (KFP v2)",
)
def data_loader_pipeline_v2(
    project_id: str = "ml-ai-portfolio",
    dataset_id: str = "taxi_forecasting",
    source_table: str = "aggregated_daily_2022",
    train_table: str = "train_2022",
    test_table: str = "test_2022",
    cutoff_date: str = "2022-11-01",
):
    _ = data_loader_component_v2(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=source_table,
        train_table=train_table,
        test_table=test_table,
        cutoff_date=cutoff_date,
    )

