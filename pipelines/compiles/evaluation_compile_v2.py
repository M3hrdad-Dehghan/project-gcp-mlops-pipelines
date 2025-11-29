import os
from kfp import compiler
from pipelines.pip.evaluation_pipeline_v2 import evaluation_pipeline_v2


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(__file__),
        "specs",
        "evaluation_pipeline_v2.json",
    )

    compiler.Compiler().compile(
        pipeline_func=evaluation_pipeline_v2,
        package_path=output_path,
    )

    print(f"Pipeline compiled successfully → {output_path}")

