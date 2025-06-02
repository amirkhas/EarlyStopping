import abc
from typing import Dict, List, Optional, Union


class RewardModel(abc.ABC):

    def __init__(self, name: str = None, **kwargs):
        super().__init__()
        self.name = name
        self.reset_usage_log()

    @abc.abstractmethod
    def score(self, prompts: List[Dict[str, str]], doc_ids: Optional[List[Union[int, str]]] = None) -> List[float]:
        pass
    
    def reset_usage_log(self):
        self._num_gen_tokens = []

    def get_usage_log_summary(self):
        return {
            "num_input_tokens": sum(self._num_input_tokens),
            "num_queries": len(self._num_input_tokens)
        }