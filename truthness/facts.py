from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from truthness.utils import CustomJSONParser, load_config


def extract(passage: str,
            model: BaseChatModel,
            language: str = "ru") -> list[str]:
    """
    Extract basic facts from the given data using the specified LLM.

    Args:
        passage (str): The data from which to extract facts.
        model (BaseChatModel): The language model to use for fact extraction.

    Returns:
        list: A list of extracted facts.
    """
    prompts = load_config("config/prompts.yaml")
    system_prompt: str = prompts.extraction.get(language)
    parser = CustomJSONParser()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=passage),
    ]

    chain = model | parser

    output = chain.invoke(messages)

    return output["facts"]
