from kfp import dsl
from pipelines.components.data_loader_component_v2 import data_loader_component_v2
from pipelines.components.train_arima_default_component_v2 import train_arima_default_component_v2
from pipelines.components.evaluation_component_v2 import evaluation_component_v2


@dsl.pipeline(
    name='taxi-forecasting-full-pipeline-v2',
    description='Load → Train ARIMA Default → Evaluate AIC (KFP v2)',
)
def full_pipeline_v2(
    project_id: str = 'ml-ai-portfolio',
    dataset_id: str = 'taxi_forecasting',
    source_table: str = 'aggregated_daily_2022',
    train_table: str = 'train_2022',
    test_table: str = 'test_2022',
    model_name: str = 'daily_arima_default_model_v1',
    cutoff_date: str = '2022-11-01',
):
    # Step 1: load
    load_task = data_loader_component_v2(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=source_table,
        train_table=train_table,
        test_table=test_table,
        cutoff_date=cutoff_date,
    )

    # Step 2: train
    train_task = train_arima_default_component_v2(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=train_table,
        model_name=model_name,
    ).after(load_task)

    # Step 3: eval
    eval_task = evaluation_component_v2(
        project_id=project_id,
        model_path=train_task.output,
    ).after(train_task)

