from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from truthness.utils import load_config


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
    prompts_hub = load_config("config/prompts.yaml")
    system_prompt: str = prompts_hub.extraction.get(language)
    parser = StrOutputParser()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=passage),
    ]

    chain = model | parser

    return chain.invoke(messages)
