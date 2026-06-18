"""CLI runner for the evaluator stage.

    uv run python -m src.pipeline.evaluator.runner \
        --index-artifact-version latest --method lightrag \
        --num-samples 20 --description "eval lightrag on graphrag-bench"

Consumes ``index_<method>``, produces ``results_<method>`` (mirrored to S3).
"""

import os
from pathlib import Path
from typing import Optional

import click

import wandb
from src.pipeline.evaluator.evaluator import Evaluator
from src.pipeline.shared.storage import generate_folder_name, get_s3_loader


@click.command()
@click.option("--index-artifact-version", required=True, type=str, help="e.g. latest, v1.")
@click.option("--method", required=True, type=str, help="Method name (e.g. lightrag).")
@click.option("--subset", default=None, type=str, help="Only evaluate this subset.")
@click.option("--source", default=None, type=str, help="Only evaluate this single corpus_name.")
@click.option(
    "--mode",
    default=None,
    type=str,
    help="Override retrieval mode (naive|local|global|hybrid|mix).",
)
@click.option("--num-samples", default=None, type=int, help="Cap questions per subset (dev).")
@click.option("--bucket", default="personal-data-science-data", type=str, help="S3 bucket.")
@click.option("--project", default="GraphRAG_Bench", type=str, help="W&B project.")
@click.option("--description", required=True, type=str, help="Experiment description.")
@click.option("--config", default="src/configs/evaluator.yaml", type=str, help="Evaluator config.")
@click.option("--providers", "providers_path", default="src/configs/providers.yaml", type=str)
@click.option("--develop", is_flag=True, default=False, help="Anonymous/offline W&B run.")
def main(
    index_artifact_version: str,
    method: str,
    subset: Optional[str],
    source: Optional[str],
    mode: Optional[str],
    num_samples: Optional[int],
    bucket: str,
    project: str,
    description: str,
    config: str,
    providers_path: str,
    develop: bool,
) -> None:
    if develop:
        os.environ.pop("WANDB_API_KEY", None)

    for arg, value in locals().items():
        print(f"{arg}: {value}")

    with wandb.init(
        project=project,
        job_type="evaluate",
        tags=["pipeline", "evaluate", method],
        notes=description,
        save_code=True,
        anonymous="must" if develop else "allow",
    ) as run:
        index_artifact = run.use_artifact(f"index_{method}:{index_artifact_version}")
        input_dir = index_artifact.download()

        artifact = wandb.Artifact(
            name=f"results_{method}", type="evaluation", description=description
        )
        output_folder_dir = Path("artifacts") / f"results_{method}" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update(
            {
                "output_dir": str(output_folder_dir),
                "source_artifact": f"index_{method}:{index_artifact_version}",
            }
        )

        s3_loader = get_s3_loader(bucket)
        evaluator = Evaluator(
            artifact,
            input_dir,
            str(output_folder_dir),
            method=method,
            config_path=config,
            providers_path=providers_path,
            subset=subset,
            source=source,
            num_samples=num_samples,
            mode=mode,
        )
        evaluator.run_evaluation()

        s3_loader.upload(output_folder_dir)
        artifact.add_reference(f"s3://{bucket}/{output_folder_dir}")
        run.log_artifact(artifact)
        print(f"\n✓ Logged results artifact 'results_{method}'. Run: {run.url}")


if __name__ == "__main__":
    main()
