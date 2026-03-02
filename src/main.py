from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import setup_logging
from src.utils.config_loader import preprocessing_cfg, model_cfg
from src.preprocessing.pipeline import PreprocessingPipeline
load_dotenv()
setup_logging()


def main() -> None:
 
    tracking_uri = model_cfg.mlflow.tracking_uri
    experiment_name = model_cfg.mlflow.experiment_name
    run_tags = dict(model_cfg.mlflow.run_tags)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")

    with mlflow.start_run(tags=run_tags) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        mlflow.log_params({
            "target_dpi": preprocessing_cfg.image.target_dpi,
            "color_mode": preprocessing_cfg.image.color_mode,
            "skew_method": preprocessing_cfg.skew_correction.method,
            "bg_removal_method": preprocessing_cfg.background_removal.method,
            "roi_method": preprocessing_cfg.roi_extraction.method,
            "roi_output_w": preprocessing_cfg.roi_extraction.output.width,
            "roi_output_h": preprocessing_cfg.roi_extraction.output.height,
        })

        raw_dir = Path("src/data/raw")
        image_paths = sorted(raw_dir.glob("*.png")) + sorted(raw_dir.glob("*.jpg")) \
                    + sorted(raw_dir.glob("*.tiff")) + sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.jpeg"))

        if not image_paths:
            logger.warning(f"No images found in {raw_dir}. Place raw cheque scans there to proceed.")
        else:
            logger.info(f"Found {len(image_paths)} raw cheque images.")
            pipeline = PreprocessingPipeline()
            results = pipeline.run_batch(image_paths, save_rois=True)

            

            succeeded = sum(1 for r in results if r.success)
            failed = len(results) - succeeded

            mlflow.log_metrics({
                "preprocessing_total": len(results),
                "preprocessing_succeeded": succeeded,
                "preprocessing_failed": failed,
            })

            logger.info(
                f"Preprocessing complete: {succeeded} succeeded, {failed} failed"
            )

        logger.info("Augmentation stage: TODO (handled by separate team member)")
        logger.info("Model training stage: TODO (handled by separate team member)")
        logger.info("Evaluation stage: TODO (handled by separate team member)")
        logger.success(f"Pipeline run complete. MLflow run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
