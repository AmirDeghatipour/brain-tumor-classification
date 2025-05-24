from src.config.configuration import ConfigurationManager
from src.components.training import PrepareCallback,Training
from src.logging import logger


STAGE_NAME = "Training Stage"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)

        training_config = config.get_training_config()
        training = Training(config=training_config, callback_handler=prepare_callbacks)
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e