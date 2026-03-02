import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru sinks:
        - stderr (coloured, INFO+)
        - logs/pipeline.log (all levels, rotating)
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler

    # Console — INFO and above
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )

    # File — DEBUG and above, rotating
    logger.add(
        log_path / "pipeline.log",
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        enqueue=True,  # Thread-safe
    )

    logger.info(f"Logging initialised → {log_path / 'pipeline.log'}")
