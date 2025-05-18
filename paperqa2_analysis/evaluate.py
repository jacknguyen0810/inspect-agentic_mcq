# Class to evaluate the performance of agent systems on multiple choice question answering

from collections.abc import Callable

import pandas as pd
from pandas import DataFrame

from inspect_ai import Epochs, Task, task, eval
from inspect_ai.agent import bridge

from paperqa2_analysis.agents.bridge_agent import bridge_agent
from paperqa2_analysis.agents.structured_agent import structured_agent
from paperqa2_analysis.inspect_ai_custom.sample import (
    record_to_sample_custom
)

from paperqa2_analysis.inspect_ai_custom.paperqa_scorer import paperqa_scorer

class MultipleChoiceEval:
    
    def __init__(
        self,
        data: DataFrame,
        agent: Callable,
        template: str | None = None,
        no_answer: str | None = None,
    ) -> None:
        
        # Process the data into inspect_ai Dataset type
        if not isinstance(data, DataFrame):
            raise TypeError("Input data is not a pandas DataFrame")

        # Check that the DataFrame has the correct columns
        req_cols = ["question", "ideal", "distractors"]
        self._check_required_columns(data, req_cols)
        
        self.dataset = data
        
        self.agent = agent
        
        self.template = template
        
        self.no_answer = no_answer
        
    
    def run(self):
        
        # Create the custom task
        @task
        def custom_agent_task():
            return Task(
                dataset=self.dataset,
                solver=bridge(
                    bridge_agent(
                        custom_agent=self.agent,
                        template=self.template
                    )
                ),
                scorer=paperqa_scorer(
                    no_answer=self.no_answer
                ),
                epochs=Epochs(1, "mode")
            )
            
        eval(custom_agent_task())
        
        return None
        
        
        
        
        
    def _check_required_columns(self, df: DataFrame, required_columns: list[str]) -> None:
        """Check if DataFrame has all required columns.
        
        Args:
            df: DataFrame to check
            required_columns: List of column names that must be present
            
        Raises:
            ValueError: If any required column is missing
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")