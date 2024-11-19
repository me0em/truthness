import json
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from truthness.metrics.base import Metric
from truthness.utils import CustomJSONParser, load_config


class IntersectionOverUnionMetric(Metric):
    """ Implementation of Intersaction over Union metric.
        rho = #I / [#(A-I) + #I + #(B-I)]
        """
    def __init__(self, njobs=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.njobs = njobs

    def __call__(self,
                 y_true: list[str],
                 y_pred: list[str]) -> float:
        chain = self.model | CustomJSONParser()

        system_prompt = self.prompts.isin_system.get(self.language)

        I_power = 0
        A_power = len(y_true)
        B_power = len(y_pred)

        def check_isin(y_hat):
            nonlocal I_power

            user_prompt = json.dumps({"f": y_hat, "A": y_true}, ensure_ascii=False)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            try:
                output = chain.invoke(messages)
                answer = output["answer"]
                if answer is True:
                    return 1
                else:
                    return 0
            except Exception as error:
                print(f"Get the error. Count this facts comparing as 0. Error: {repr(error)}")
                return 0

        with ThreadPoolExecutor(max_workers=self.njobs) as executor:
            results = list(executor.map(check_isin, y_pred))

        I_power = sum(results)

        score = I_power / (A_power + B_power - I_power)

        return score