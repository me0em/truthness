from typing import Optional, UUID
from uuid import uuid4
import json
import glob
from pydantic import BaseModel

from langchain_core.language_models import BaseChatModel

from truthness.facts import extract
from truthness.utils import get_model, get_golden_passages, get_datasets_for_eval
from truthness.metrics import Metric, IntersectionOverCartesianMetric


class ExperimentJournalModel(BaseModel):
    experiment_id: UUID
    eval_model: str
    scores: list[str]
    mean: float
    golden_facts_mapper: dict[str, list[str]]
    eval_facts_mapper: dict[str, list[str]]


def expirement(golden_passages: list[str],
               eval_passages: list[str],
               eval_model: str,
               metric: Metric,
               model: BaseChatModel,
               eval_facts_mapper: dict[str, list[str]] | dict = {},
               golden_facts_mapper: dict[str, list[str]] | dict = {}) -> BaseModel:
    """ TODO: doc string
    """
    experiment_id = uuid4().hex

    print(f"Start experiment {experiment_id}")

    scores = []
    for idx in range(len(golden_passages)):
        gld_passage = golden_passages[idx]
        eval_passage = eval_passages[idx]

        if gld_passage not in golden_facts_mapper:
            golden_facts: list[str] = extract(model=model,
                                              language="ru",
                                              passage=gld_passage)
            golden_facts_mapper[gld_passage] = golden_facts

        if eval_passage not in eval_facts_mapper:
            eval_facts: list[str] = extract(model=model,
                                            language="ru",
                                            passage=eval_passage)
            eval_facts_mapper[eval_passage] = eval_facts

        score: float = metric(
            y_true=golden_facts,
            y_pred=eval_facts
        )

        print("golden facts", golden_facts)
        print("evaluated facts", eval_facts)
        print("score", score)

        scores.append(score)

    journal[eval_model] = {
        "scores": scores,
        "mean": sum(scores) / len(scores)
    }

    journal = ExperimentJournalModel(
        experiment_id=experiment_id,
        eval_model=eval_model,
        scores=scores,
        mean=sum(scores)/(len(scores)+1e-6),
        golden_facts_mapper=golden_facts_mapper,
        eval_facts_mapper=eval_facts_mapper
    )

    return journal     

if __name__ == "__main__":
    model = get_model(model_name="gpt-4o-mini")
    ioc = IntersectionOverCartesianMetric(model=model, language="ru")
    golden_passages = get_golden_passages()
    datasets = get_datasets_for_eval()

    journals = {}

    for eval_model, eval_passages in tqdm(datasets.items()):
        journal = expirement(
            golden_passages=golden_passages,
            eval_passages=eval_passages,
            eval_model=eval_model,
            metric=ioc,
            model=model,
            eval_facts_mapper={}
            golden_facts_mapper={}
        )
    
        os.makedirs(f"journals/{eval_model}", exist_ok=True)
        with open(f"journals/{eval_model}/{journal.experiment_id}.json", "w") as file:
            json.dump(journal.json(), file)
