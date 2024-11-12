import json

from langchain_core.messages import HumanMessage, SystemMessage

from truthness.metrics.base import Metric
from truthness.utils import CustomJSONParser, load_config


class IntersectionOverCartesianMetric(Metric):
    """ Implementation of Intersaction over Cartesian metric.
        This metric is straight alternative to classic IoU.
        Each passage, i.e. answer from golden dataset,
        and RAG generation, are decomposes on set of facts,
        then we calculate proportion of similar pairs
        in the all pairs.
        """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self,
                 y_true: list[str],
                 y_pred: list[str]) -> float:
        chain = self.model | CustomJSONParser()
        system_prompt = self.prompts.facts_equal.get(self.language)

        correctness = []
        for y in y_true:
            for y_hat in y_pred:
                facts = {"fact_1": y, "fact_2": y_hat}
                facts = json.dumps(facts, ensure_ascii=False)

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=facts),
                ]

                try:
                    output = chain.invoke(messages)
                    answer = output["answer"]
                    correctness.append(answer)
                except Exception as error:
                    print(f"Get the error. Count this facts comparing as 0. Error: {repr(error)}")
                    correctness.append(0)

        score = sum(correctness) / (len(correctness) + 1e-6)

        return score
