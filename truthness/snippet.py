from truthness.facts import extract
from truthness.utils import get_model
from truthness.metrics import IntersectionOverUnionMetric

model = get_model(model_name="gpt-4o-mini")


# mocked_ground_truth_passage = "Трамп выиграл выборы в 2024 году, хотя многие думали что выиграет Харрис, большинство людей не знала что избирается кто-то ещё, хотя было ещё три человека от партий вне двухпартийной системы."
# mocked_models_passage = "Трамп выиграл выборы в 2024, он был удинственным кандидатом"

mocked_ground_truth_passage = "Трамп выиграл выборы"
mocked_models_passage = "выборы выиграл трамп"

ground_truth_facts: list[str] = extract(
    model=model,
    language="ru",
    passage=mocked_ground_truth_passage
)

models_facts: list[str] = extract(
    model=model,
    language="ru",
    passage=mocked_models_passage
)

print(f"• Ground-truth Facts: {ground_truth_facts}")
print(f"• Model's Facts: {models_facts}")

iou = IntersectionOverUnionMetric(model=model, language="ru")

score: float = iou(
    y_true=ground_truth_facts,
    y_pred=models_facts
)

print(score)
