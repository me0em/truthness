from truthness.facts import extract
from truthness.utils import get_model

model = get_model(model_name="gpt-4o-mini")

facts: list[str] = extract(
    model=model,
    language="ru",
    passage="Трамп выиграл выборы в 2024 году, хотя многие думали что выиграет Харрис, большинство людей не знала что избирается кто-то ещё, хотя было ещё три человека от партий вне двухпартийной системы."
)

print(facts)