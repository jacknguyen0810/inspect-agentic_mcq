from typing import Any, List
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, metric
from inspect_ai.solver import TaskState

from inspect_evals.lab_bench.metrics import coverage, precision


# @metric
# def precision(scores: List[Score]) -> float:
#     """Precision metric for PaperQA evaluations."""
#     if not scores:
#         return 0.0
#     return sum(s.metrics.get("precision", 0) for s in scores) / len(scores)


@scorer(metrics=[accuracy(), precision()])
def paperqa_scorer(no_answer: str = "Insufficient information to answer the question.") -> Scorer:
    """
    Custom scorer for PaperQA outputs that calculates accuracy and precision.
    
    Args:
        no_answer: String indicating the model doesn't have enough information
        
    Returns:
        A scoring function compatible with inspect_ai
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Get the output from the bridged function
        output = state.output
        
        # Get the target answer
        target_answer = target.text
        
        # Calculate metrics
        is_correct = output == target_answer
        is_no_answer_target = target_answer == no_answer
        is_no_answer_output = output == no_answer
        
        # Compute metrics
        if is_no_answer_target and is_no_answer_output:
            # Correctly identified no answer
            result = "correct"
            metrics = {"accuracy": 1.0, "precision": 1.0}
        elif is_no_answer_target and not is_no_answer_output:
            # False positive
            result = "incorrect"
            metrics = {"accuracy": 0.0, "precision": 0.0}
        elif not is_no_answer_target and is_no_answer_output:
            # False negative
            result = "incorrect"
            metrics = {"accuracy": 0.0, "precision": 1.0}
        else:
            # Regular answer case
            result = "correct" if is_correct else "incorrect"
            metrics = {
                "accuracy": 1.0 if is_correct else 0.0,
                "precision": 1.0 if is_correct else 0.0
            }
        
        return Score(
            value=result,
            answer=output,
            explanation=state.output,
            metrics=metrics
        )
    
    return score


# Usage example with a task
"""
from inspect_ai import task, Task, Epochs
from inspect_ai.agent import bridge
from paperqa2_analysis.scorers.paperqa_scorer import paperqa_scorer
from paperqa2_analysis.agents.paperqa_agent import paperqa_agent, MULTIPLE_CHOICE_TEMPLATE_BRIDGE, paperqa_settings

@task
def paperqa_eval_task():
    return Task(
        dataset=test_dataset,
        solver=bridge(paperqa_agent(
            llm_config=llm_config,
            settings=paperqa_settings,
            template=MULTIPLE_CHOICE_TEMPLATE_BRIDGE
        )),
        scorer=paperqa_scorer(),
        epochs=Epochs(1)
    )
"""

if __name__ == "__main__":
    # Test the scorer with a mock state and target
    class MockState:
        class MockOutput:
            completion = "A"
            content = "The answer is A"
        output = MockOutput()
    
    class MockTarget:
        text = "A"
    
    import asyncio
    
    async def test_scorer():
        scorer_fn = paperqa_scorer()
        result = await scorer_fn(MockState(), MockTarget())
        print(f"Result: {result.value}")
        print(f"Metrics: {result.metrics}")
    
    asyncio.run(test_scorer()) 