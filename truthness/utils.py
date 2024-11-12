import yaml
from box import Box

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def load_config(path: str) -> Box:
    with open(path, "r") as file:
        yaml_config = yaml.safe_load(file)

    return Box(yaml_config)


def get_model(model_name: str) -> BaseChatModel:
    match model_name:
        case "gpt-4o-mini":
            return ChatOpenAI(model=model_name)
        case "gpt-4o":
            return ChatOpenAI(model=model_name)
        case "gpt-3.5-turbo":
            return ChatOpenAI(model=model_name)

        case _:
            raise NotImplementedError(
                f"Model {model_name} not implemented."
                 "You can add it by yourself, check utils.get_model"
            )
