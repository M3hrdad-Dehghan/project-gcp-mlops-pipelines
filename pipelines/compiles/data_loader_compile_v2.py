import os
from kfp import compiler
from pipelines.pip.data_loader_pipeline_v2 import data_loader_pipeline_v2


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(__file__),
        "specs",
        "data_loader_pipeline_v2.json",
    )

    compiler.Compiler().compile(
        pipeline_func=data_loader_pipeline_v2,
        package_path=output_path,
    )

    print(f"Pipeline compiled successfully → {output_path}")

