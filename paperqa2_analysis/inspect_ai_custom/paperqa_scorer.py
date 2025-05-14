import json

from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, metric
from inspect_ai.solver import TaskState

@scorer(metrics=accuracy())
def paperqa_scorer(no_answer: str) -> Scorer:
    
    # Create async score function
    async def score(state: TaskState) -> Score:
        
        # use json to load the answer
        output = json.loads(state.output)
        
        answer = output["answer"]
        target = output["target"]
        
        # Calculate metrics
        is_correct = output == target
        is_no_answer_target = target == no_answer
        is_no_answer_output = output == no_answer
        
        