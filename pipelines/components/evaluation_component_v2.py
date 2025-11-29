from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"],
)
def evaluation_component_v2(
    project_id: str,
    model_path: str,
) -> float:
    """
    Evaluate the trained BQML ARIMA model and return the AIC score (float).
    This component tries to import the project's `logic_components` package; in case
    it's not available in the runtime container, it provides a fallback implementation.
    """
    import traceback

    try:
        try:
            from logic_components.model_evaluate import ModelEvaluator
        except Exception:
            # fallback implementation using google-cloud-bigquery
            from google.cloud import bigquery
            import pandas as pd

            class ModelEvaluator:
                def __init__(self, project_id: str) -> None:
                    self.project_id = project_id
                    self.client = bigquery.Client(project=project_id)

                def evaluate(self, model_path: str) -> dict:
                    # Use the job.to_dataframe() helper if available
                    query = f"""
                    SELECT aic
                    FROM ML.ARIMA_EVALUATE(MODEL `{model_path}`)
                    LIMIT 1
                    """
                    job = self.client.query(query)
                    try:
                        df = job.to_dataframe()
                    except Exception:
                        df = job.result().to_dataframe()

                    if df.empty:
                        raise RuntimeError(f"No evaluation results found for model: {model_path}")
                    aic_value = float(df["aic"].iloc[0])
                    return {"model": model_path, "aic": aic_value}

        evaluator = ModelEvaluator(project_id=project_id)
        print(f"[Eval] Evaluating model: {model_path}")
        result = evaluator.evaluate(model_path=model_path)
        # Try to give an early hint if model exists in BQ (useful for diagnosing permission/missing model)
        try:
            parts = model_path.split('.')
            if len(parts) == 3:
                p, d, m = parts
                check_query = f"SELECT COUNT(*) as cnt FROM `{p}.{d}.INFORMATION_SCHEMA.MODELS` WHERE model_name = '{m}'"
                check_job = evaluator.client.query(check_query)
                try:
                    check_df = check_job.to_dataframe()
                except Exception:
                    check_df = check_job.result().to_dataframe()
                has_model = int(check_df['cnt'].iloc[0]) > 0
                print(f"[Eval] model_exists: {has_model} for {model_path}")
            else:
                print(f"[Eval] Could not parse model_path for existence check: {model_path}")
        except Exception as e:
            print(f"[Eval] Error while checking model existence: {e}")
        aic_value = float(result.get("aic"))
        print(f"AIC: {aic_value}")
        return aic_value
    except Exception as e:
        # Print stacktrace to logs for debug; re-raise so pipeline fails clearly
        print("[Eval] Error during evaluation:")
        traceback.print_exc()
        raise

