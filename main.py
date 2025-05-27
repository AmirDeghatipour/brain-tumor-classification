from src.logging import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.pipeline.stage_03_training import ModelTrainingPipeline
from src.pipeline.stage_04_model_evaluation import ModelEvaluationPipepile


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Training Model Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipepile()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e