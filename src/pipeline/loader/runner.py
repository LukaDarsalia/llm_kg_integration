"""CLI runner for the loader stage.

    uv run python -m src.pipeline.loader.runner --description "load graphrag-bench"

Produces the W&B artifact ``dataset`` (mirrored to S3 by reference).
"""

import os
from pathlib import Path

import click
import wandb

from src.pipeline.loader.loader import DatasetLoader
from src.pipeline.shared.storage import generate_folder_name, get_s3_loader


@click.command()
@click.option("--bucket", default="personal-data-science-data", type=str, help="S3 bucket for data.")
@click.option("--project", default="GraphRAG_Bench", type=str, help="W&B project name.")
@click.option("--description", required=True, type=str, help="Experiment description.")
@click.option("--config", default="src/configs/loader.yaml", type=str, help="Loader config.")
@click.option("--develop", is_flag=True, default=False, help="Anonymous/offline W&B run.")
def main(bucket: str, project: str, description: str, config: str, develop: bool) -> None:
    if develop:
        os.environ.pop("WANDB_API_KEY", None)

    for arg, value in locals().items():
        print(f"{arg}: {value}")

    with wandb.init(
        project=project,
        job_type="load-data",
        tags=["pipeline", "dataset"],
        notes=description,
        save_code=True,
        anonymous="must" if develop else "allow",
    ) as run:
        artifact = wandb.Artifact(name="dataset", type="dataset", description=description)
        output_folder_dir = Path("artifacts") / "dataset" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({"output_dir": str(output_folder_dir)})

        s3_loader = get_s3_loader(bucket)
        loader = DatasetLoader(artifact, str(output_folder_dir), config_path=config)

        print("\nDataset configuration:")
        for name, info in loader.get_dataset_info().items():
            status = "✓ Enabled" if info["enabled"] else "✗ Disabled"
            print(f"  {name}: {status} - {info['description']}")

        loader.run_loading()

        s3_loader.upload(output_folder_dir)
        artifact.add_reference(f"s3://{bucket}/{output_folder_dir}")
        run.log_artifact(artifact)
        print(f"\n✓ Logged dataset artifact. Run: {run.url}")


if __name__ == "__main__":
    main()
