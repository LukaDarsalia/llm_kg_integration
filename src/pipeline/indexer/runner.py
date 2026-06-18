"""CLI runner for the indexer stage.

    uv run python -m src.pipeline.indexer.runner \
        --dataset-artifact-version latest --method lightrag \
        --description "lightrag index of graphrag-bench"

Consumes the ``dataset`` artifact, produces ``index_<method>`` (mirrored to S3).
"""

import os
from pathlib import Path
from typing import Optional

import click

import wandb
from src.pipeline.indexer.indexer import Indexer
from src.pipeline.shared.storage import generate_folder_name, get_s3_loader


@click.command()
@click.option("--dataset-artifact-version", required=True, type=str, help="e.g. latest, v1.")
@click.option("--method", required=True, type=str, help="Method name (e.g. lightrag).")
@click.option("--subset", default=None, type=str, help="Only index this subset (default: all).")
@click.option(
    "--source", default=None, type=str, help="Only index this single corpus_name (e.g. one novel)."
)
@click.option("--bucket", default="personal-data-science-data", type=str, help="S3 bucket.")
@click.option("--project", default="GraphRAG_Bench", type=str, help="W&B project.")
@click.option("--description", required=True, type=str, help="Experiment description.")
@click.option(
    "--config",
    default=None,
    type=str,
    help="Method params yaml (default: src/configs/<method>_params.yaml).",
)
@click.option("--providers", "providers_path", default="src/configs/providers.yaml", type=str)
@click.option("--develop", is_flag=True, default=False, help="Anonymous/offline W&B run.")
def main(
    dataset_artifact_version: str,
    method: str,
    subset: Optional[str],
    source: Optional[str],
    bucket: str,
    project: str,
    description: str,
    config: Optional[str],
    providers_path: str,
    develop: bool,
) -> None:
    if develop:
        os.environ.pop("WANDB_API_KEY", None)

    config = config or f"src/configs/{method}_params.yaml"

    for arg, value in locals().items():
        print(f"{arg}: {value}")

    with wandb.init(
        project=project,
        job_type="index",
        tags=["pipeline", "index", method],
        notes=description,
        save_code=True,
        anonymous="must" if develop else "allow",
    ) as run:
        dataset_artifact = run.use_artifact(f"dataset:{dataset_artifact_version}")
        input_dir = dataset_artifact.download()

        artifact = wandb.Artifact(name=f"index_{method}", type="index", description=description)
        output_folder_dir = Path("artifacts") / f"index_{method}" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update(
            {
                "output_dir": str(output_folder_dir),
                "source_artifact": f"dataset:{dataset_artifact_version}",
            }
        )

        s3_loader = get_s3_loader(bucket)
        indexer = Indexer(
            artifact,
            input_dir,
            str(output_folder_dir),
            method=method,
            config_path=config,
            providers_path=providers_path,
            subset=subset,
            source=source,
        )
        indexer.run_indexing()

        s3_loader.upload(output_folder_dir)
        artifact.add_reference(f"s3://{bucket}/{output_folder_dir}")
        run.log_artifact(artifact)
        print(f"\n✓ Logged index artifact 'index_{method}'. Run: {run.url}")


if __name__ == "__main__":
    main()
