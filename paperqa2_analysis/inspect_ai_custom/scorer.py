from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)

from inspect_ai.solver import TaskState
@scorer(metrics=[accuracy(), precision(), coverage()])
def precision_choice_custom(no_answer: str | None = None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        choices = state.choices
        explanation = state.output.completion
        
        






