import os
from kfp import compiler
from pipelines.full_pipeline_v2 import full_pipeline_v2


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(__file__),
        "specs",
        "full_pipeline_v2.json",
    )

    compiler.Compiler().compile(
        pipeline_func=full_pipeline_v2,
        package_path=output_path,
    )

    print(f"Full pipeline compiled successfully → {output_path}")

