from langchain_core.language_models import BaseChatModel

from truthness.utils import load_config


class Metric:
    """ Base class for different metric
    Objects has similar to sklearn.metrics interface
    """
    def __init__(self,
                 model: BaseChatModel,
                 language: str) -> None:
        self.model = model
        self.language = language
        self.prompts = load_config("config/prompts.yaml")

    def __self__(y_true: str, y_pred: str) -> float:
        raise NotImplementedError
