import json
import glob

from truthness.facts import extract
from truthness.utils import get_model, get_golden_passages, get_datasets_for_eval
from truthness.metrics import IntersectionOverCartesianMetric


if __name__ == "__main__":
    model = get_model(model_name="gpt-4o-mini")
    ioc = IntersectionOverCartesianMetric(model=model, language="ru")
    golden_passages = get_golden_passages()
    datasets = get_datasets_for_eval()
    journal = {}

    for evl_model, evl_passages in datasets.items():
        scores = []
        
        print(f"Start score {evl_model} versus Golden")

        for gld_passage, evl_passage in zip(golden_passages, evl_passages):
            golden_facts: list[str] = extract(
                model=model,
                language="ru",
                passage=gld_passage
            )

            evl_facts: list[str] = extract(
                model=model,
                language="ru",
                passage=evl_passage
            )

            score: float = ioc(
                y_true=golden_facts,
                y_pred=evl_facts
            )

            print("golden facts", golden_facts)
            print("evaluated facts", evl_facts)
            print("score", score)

            scores.append(score)

        journal[evl_model] = {
            "scores": scores,
            "mean": sum(scores) / len(scores)
        }

        logs_path = f"results/{evl_model}.json"
        with open(logs_path, "w") as file:
            json.dump(
                {evl_model: journal[evl_model]},
                file
            )
        print(f"Dumps {evl_model} in {logs_path} with score {journal[evl_model]['mean']}")
            
