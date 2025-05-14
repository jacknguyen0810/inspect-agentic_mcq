import json

from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, metric, SampleScore, ValueToFloat, Value, value_to_float, CORRECT, INCORRECT, NOANSWER, Metric
from inspect_ai.solver import TaskState

      

# Custom Value to Float function
def precision_value_to_float(
    correct: Value = CORRECT,
    incorrect: Value = INCORRECT,
    noanswer: Value = NOANSWER,
) -> ValueToFloat:
    def to_float(value: Value) -> float:
        # Catch non-string values
        if isinstance(value, int | float | bool):
            return float(value)
        # Returh -1 score for unable to answer.
        elif value == noanswer:  
            return -1
        else:
            return value_to_float(
                correct=correct,
                incorrect=incorrect,
                noanswer=noanswer
            )(value)
    return to_float
        

# Custom Metrics
@metric
def paperqa_precision(to_float: ValueToFloat = precision_value_to_float()) -> Metric:
    
    def metric(scores: list[SampleScore]) -> float:
        # Get the answered questions
        answered = [i for i in scores if to_float(i.score.value) != -1]
        # Check if no questions answered
        if len(answered) == 0:
            return 0.0
        
        total = 0.0
        for i in answered:
            total += to_float(i.score.value)
        return total / float(len(answered))
    
    return metric

@metric
def paperqa_accuracy(to_float: ValueToFloat = precision_value_to_float()) -> Metric:
    
    def metric(scores: list[SampleScore]) -> float:     
        total = 0.0
        for i in scores:
            total += to_float(i.score.value)
        return total / float(len(scores))
    
    return metric



@scorer(metrics=[paperqa_accuracy(), paperqa_precision()])
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