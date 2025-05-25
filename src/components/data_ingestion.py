from src.config.configuration import DataIngestionConfig
from src.logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Checking Dataset Directory is exsisting")
        if not self.config.source_data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found at {self.config.source_data_dir}")
        logger.info("Dataset directory found and ready to use")
        return self.config.source_data_dir
