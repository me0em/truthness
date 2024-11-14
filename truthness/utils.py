import os
import json
import re
import glob

import yaml
from box import Box

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter


def load_config(path: str) -> Box:
    with open(path, "r") as file:
        yaml_config = yaml.safe_load(file)

    return Box(yaml_config)


class CustomJSONParser(StrOutputParser):
    def parse(self, text: str) -> dict:
        # Regular expression to find JSON content within triple backticks
        json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        match = json_pattern.search(text)

        if match:
            json_str = match.group(1).strip()
            try:
                # Parse the JSON string into a dictionary
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON content: {json_str}") from e
        else:
            raise ValueError("No JSON content found in the text")


def get_model(model_name: str) -> BaseChatModel:
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=5,  # 1 req every 200ms
        check_every_n_seconds=0.1  # check every 100ms
    )

    match model_name:
        case "gpt-4o-mini":
            return ChatOpenAI(model=model_name, rate_limiter=rate_limiter)
        case "gpt-4o":
            return ChatOpenAI(model=model_name, rate_limiter=rate_limiter)
        case "gpt-3.5-turbo":
            return ChatOpenAI(model=model_name, rate_limiter=rate_limiter)

        case _:
            raise NotImplementedError(
                f"Model {model_name} not implemented."
                 "You can add it by yourself, check utils.get_model"
            )


def get_golden_passages(path="data/golden.json") -> list[str]:
    if not os.path.isfile(path):
        raise Exception(f"Golden dataset is not founded at {path}")

    with open(path, "r") as file:
        golden_passages = json.load(file)["passages"]
    
    return golden_passages


def get_datasets_for_eval() -> dict[str, list[str]]:
    paths = glob.glob("data/*.json")
    paths = [path for path in paths if path.split("/")[-1] != "golden.json"]

    datasets = {}

    for path in paths:
        with open(path, "r") as file:
            dataset = json.load(file)
            datasets[dataset["actor"]] = dataset["passages"]

    return datasets
