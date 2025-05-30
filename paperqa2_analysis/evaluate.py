# Class to evaluate the performance of agent systems on multiple choice question answering

from collections.abc import Callable
import inspect

from pandas import DataFrame

from inspect_ai import Epochs, Task, task, eval
from inspect_ai.agent import bridge

from paperqa2_analysis.agents.bridge_agent import bridge_agent
from paperqa2_analysis.inspect_ai_custom.sample import df_2_sample_bridge

from paperqa2_analysis.inspect_ai_custom.paperqa_scorer import paperqa_scorer


class MultipleChoiceEval:
    """Class for evaluating MCQ performance for inspect_ai using custom agents."""

    def __init__(
        self, data: DataFrame, agent: Callable, template: str | None = None, **kwargs
    ) -> None:

        # Process the data into inspect_ai Dataset type
        if not isinstance(data, DataFrame):
            raise TypeError("Input data is not a pandas DataFrame")

        # Check that the DataFrame has the correct columns
        req_cols = ["question", "ideal", "distractors"]
        self._check_required_columns(data, req_cols)

        # Check that the agent is valid
        self._validate_custom_agent(agent)

        # Set up the dataset
        self.dataset = df_2_sample_bridge(data)

        self.agent = agent
        self.template = template
        self.kwargs = kwargs
        
        # Cost and token usage
        self.cost = 0.0
        self.token_counts = {}

    def run(
        self,
        max_samples: int | None,
        time_limit: float | None,
    ):
        """Run the inspect_ai benchmarking.

        Returns:
            dict: Dictionary containing evaluation results, total cost, and token usage.
        """
        # Create the custom task
        @task
        def custom_agent_task():
            return Task(
                dataset=self.dataset,
                solver=bridge(
                    bridge_agent(
                        custom_agent=self.agent, template=self.template, **self.kwargs
                    )
                ),
                scorer=paperqa_scorer(),
                epochs=Epochs(1, "mode"),
            )

        # Run eval and collect outputs for cost/token usage
        eval_result = eval(tasks=custom_agent_task(), time_limit=time_limit, max_samples=max_samples)
        
        # Initialize tracking variables
        total_cost = 0.0
        total_token_counts = {}
        
        # Process results if eval_result is iterable
        if hasattr(eval_result, '__iter__'):
            for r in eval_result:
                # Get cost and token counts from the eval log
                if hasattr(r, 'eval') and hasattr(r.eval, 'cost'):
                    total_cost += float(r.eval.cost)
                
                if hasattr(r, 'eval') and hasattr(r.eval, 'token_counts'):
                    for model, counts in r.eval.token_counts.items():
                        if model not in total_token_counts:
                            total_token_counts[model] = [0, 0]  # [prompt_tokens, completion_tokens]
                        if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                            total_token_counts[model][0] += int(counts[0])
                            total_token_counts[model][1] += int(counts[1])
        
        # Update instance variables
        self.cost = total_cost
        self.token_counts = total_token_counts
        
        # Print summary
        print("\n--- Evaluation Cost Summary ---")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Total token usage: {total_token_counts}")
        print("------------------------------\n")
        
        # Return results
        return {
            "cost": total_cost,
            "token_counts": total_token_counts,
            "eval_result": eval_result
        }

    def _check_required_columns(
        self, df: DataFrame, required_columns: list[str]
    ) -> None:
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

    def _validate_custom_agent(self, custom_agent: Callable) -> None:
        """Validate that the custom agent is a function that takes in and returns a dictionary with answer, cost, and token counts.

        Args:
            custom_agent: The custom agent function to validate

        Raises:
            TypeError: If the custom agent is not a function or doesn't have the correct signature
        """

        if not callable(custom_agent):
            raise TypeError("Custom agent must be a callable function")

        # Get the function signature
        sig = inspect.signature(custom_agent)

        # Check that the function takes at least one parameter
        if len(sig.parameters) < 1:
            raise TypeError(
                f"Custom agent must take at least one parameter, got {len(sig.parameters)}"
            )

        # Check the return type annotation if it exists
        return_annotation = sig.return_annotation
        if return_annotation != inspect.Signature.empty and return_annotation != dict:
            raise TypeError(
                f"Custom agent must return a dict with 'answer', 'cost', and 'token_counts' keys, got return type annotation {return_annotation}"
            )
