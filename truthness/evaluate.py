import json
import glob

from truthness.facts import extract
from truthness.utils import get_model
from truthness.metrics import IntersectionOverCartesianMetric

model = get_model(model_name="gpt-4o-mini")

paths = glob.glob("data/*.json")
print(paths)
ioc = IntersectionOverCartesianMetric(model=model, language="ru")

datasets = {}
for path in paths:
    with open(path, "r") as file:
        dataset = json.load(file)
        datasets[dataset["actor"]] = dataset["passages"]

golden_passages = datasets["golden"]
del datasets["golden"]

journal = {}

for scored_model, scored_passages in datasets.items():

    scores = []
    
    print(f"Start score {scored_model} versus golden")

    for golden_passage, scored_passage in zip(golden_passages,
                                              scored_passages):
        golden_facts: list[str] = extract(
            model=model,
            language="ru",
            passage=golden_passage
        )

        models_facts: list[str] = extract(
            model=model,
            language="ru",
            passage=scored_passage
        )

        score: float = ioc(
            y_true=golden_facts,
            y_pred=models_facts
        )

        print("golden facts", golden_facts)
        print("models facts", models_facts)
        print("score", score)

        scores.append(score)

    journal[scored_model] = {
        "scores": scores,
        "mean": sum(scores) / len(scores)
    }

    logs_path = f"results/{scored_model}.json"
    with open(logs_path, "w") as file:
        json.dump(
            {scored_model: journal[scored_model]},
            file
        )
    print(f"Dumps {scored_model} in {logs_path} with score {journal[scored_model]['mean']}")
        
