import json
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from truthness.metrics.base import Metric
from truthness.utils import CustomJSONParser, load_config

from truthness.metrics import RecallMetric
from truthness.metrics import PrecisionMetric


class MultipleMetric(Metric):
    def __init__(self, njobs=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.njobs = njobs

        self.precision_metric = PrecisionMetric(
            self.njobs,
            *args,
            **kwargs
        )

        self.recall_metric = RecallMetric(
            self.njobs,
            *args,
            **kwargs
        )

    def __call__(self,
                 y_true: list[str],
                 y_pred: list[str]) -> float:

        rcl = self.recall_metric(y_true, y_pred)
        prc = self.precision_metric(y_true, y_pred)
        f1 = 2 * prc * rcl / (prc + rcl)

        scores = {
            "recall": rcl,
            "precision": prc,
            "f1": f1
        }

        print(scores)

        return scores
