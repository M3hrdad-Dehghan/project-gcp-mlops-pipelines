import os
from kfp import compiler
from pipelines.pip.train_arima_default_pipeline_v2 import train_arima_default_pipeline_v2


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(__file__),
        "specs",
        "train_arima_default_pipeline_v2.json",
    )

    compiler.Compiler().compile(
        pipeline_func=train_arima_default_pipeline_v2,
        package_path=output_path,
    )

    print(f"Pipeline compiled successfully → {output_path}")

