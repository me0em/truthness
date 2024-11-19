from typing import Optional
import os
from uuid import uuid4, UUID
import json
import glob
from pydantic import BaseModel
from tqdm import tqdm

from langchain_core.language_models import BaseChatModel

from truthness.facts import extract
from truthness.utils import get_model, get_golden_passages, get_datasets_for_eval
from truthness.metrics import Metric, IntersectionOverUnionMetric


class ExperimentJournalModel(BaseModel):
    experiment_id: UUID
    eval_model: str
    scores: list[float]
    mean: float
    golden_facts_mapper: dict[str, list[str]]
    eval_facts_mapper: dict[str, list[str]]
    scored_by_golden_passage_mapper: dict[str, float] | dict
    size: int
    failed_counter: int


def expirement(golden_passages: list[str],
               eval_passages: list[str],
               eval_model: str,
               metric: Metric,
               model: BaseChatModel,
               scored_by_golden_passage_mapper: dict[str, float] | dict,
               eval_facts_mapper: dict[str, list[str]] | dict = {},
               golden_facts_mapper: dict[str, list[str]] | dict = {}) -> BaseModel:
    """ TODO: doc string
    """
    experiment_id = uuid4().hex

    scores = []
    failed_counter = 0
    keyboard_interrupt_flag = False
    for idx in tqdm(range(len(golden_passages)),
                    total=len(golden_passages),
                    desc=eval_model):

        if keyboard_interrupt_flag:
            break
        
        try:
            gld_passage = golden_passages[idx]
            eval_passage = eval_passages[idx]

            if gld_passage in scored_by_golden_passage_mapper:
                # we already scored this passage
                continue

            if gld_passage not in golden_facts_mapper:
                golden_facts: list[str] = extract(model=model,
                                                language="ru",
                                                passage=gld_passage)
                golden_facts_mapper[gld_passage] = golden_facts
            else:
                golden_facts = golden_facts_mapper[gld_passage]


            if eval_passage not in eval_facts_mapper:
                eval_facts: list[str] = extract(model=model,
                                                language="ru",
                                                passage=eval_passage)
                eval_facts_mapper[eval_passage] = eval_facts
            else:
                eval_facts = eval_facts_mapper[eval_passage]

            score: float = metric(
                y_true=golden_facts,
                y_pred=eval_facts
            )

            scored_by_golden_passage_mapper[gld_passage] = score

            scores.append(score)

        except KeyboardInterrupt:
            keyboard_interrupt_flag = True

        except Exception as error:
            print(f"Error: {error}")
            failed_counter += 1

    journal = ExperimentJournalModel(
        experiment_id=experiment_id,
        eval_model=eval_model,
        scores=scores,
        mean=sum(scores)/(len(scores)+1e-6),
        golden_facts_mapper=golden_facts_mapper,
        eval_facts_mapper=eval_facts_mapper,
        scored_by_golden_passage_mapper=scored_by_golden_passage_mapper,
        size=len(golden_passages),
        failed_counter=failed_counter
    )

    return journal     

if __name__ == "__main__":
    model = get_model(model_name="gpt-4o")
    iou = IntersectionOverUnionMetric(
        model=model,
        language="ru",
        njobs=3
    )
    golden_passages = get_golden_passages()
    datasets = get_datasets_for_eval()

    # datasets = {k:v for k,v in datasets.items() if k=="nemo-unsloth-248-x8"}
    # datasets = {k:v for k,v in datasets.items() if k=="gpt-4o"}

    with open("journals/200/4c2f9a23-43e1-490f-bca7-528236497a4c.json", "r") as file:
        d = json.load(file)
        d = eval(d)
        golden_facts_mapper = d["golden_facts_mapper"]

    with open("journals/600/70f08bd2-637e-4a78-b037-3b0315eed66c.json", "r") as file:
        d = json.load(file)
        d = eval(d)
        scored_by_golden_passage_mapper = d["scored_by_golden_passage_mapper"]

    journals = {}

    for eval_model, eval_passages in tqdm(datasets.items(),
                                          total=len(datasets),
                                          desc="Score models"):
        journal = expirement(
            golden_passages=golden_passages,
            eval_passages=eval_passages,
            eval_model=eval_model,
            metric=iou,
            model=model,
            scored_by_golden_passage_mapper=scored_by_golden_passage_mapper,
            eval_facts_mapper={},
            golden_facts_mapper=golden_facts_mapper
        )
    
        os.makedirs(f"journals/{eval_model}", exist_ok=True)
        with open(f"journals/{eval_model}/{journal.experiment_id}.json", "w") as file:
            json.dump(journal.json(), file, ensure_ascii=False)
