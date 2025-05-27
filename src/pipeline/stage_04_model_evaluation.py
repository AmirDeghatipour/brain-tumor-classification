from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import PrepareCallback, Evaluation
from src.logging import logger


STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipepile:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        evaluation = Evaluation(val_config, prepare_callbacks)
        evaluation.evaluate()
        evaluation.save_score()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipepile()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e