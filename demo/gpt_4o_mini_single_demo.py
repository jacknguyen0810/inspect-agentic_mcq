import pandas as pd
from pandas import DataFrame

from inspect_ai import Task, Epochs, eval, task
from inspect_ai.solver import multiple_choice

from inspect_evals.lab_bench.scorer import precision_choice

from paperqa2_analysis.inspect_ai_custom.sample import df_2_sample

if __name__ == "__main__":
    # Get the dataset
    # Import data
    litqa2_test_data = pd.read_parquet("/root/paperQA2_analysis/data/LitQA_data/test-00000-of-00001.parquet")
    
    # Create mini dataset with one sample
    mini_data = pd.DataFrame([{
        "question": litqa2_test_data["question"][0],
        "ideal": litqa2_test_data["ideal"][0],
        "distractors": litqa2_test_data["distractors"][0]
    }])
    
    mini_data = pd.DataFrame(
        [
            {
                "question": litqa2_test_data["question"][0],
                "ideal": litqa2_test_data["ideal"][0],
                "distractors": litqa2_test_data["distractors"][0]
            },
            {
                "question": litqa2_test_data["question"][1],
                "ideal": litqa2_test_data["ideal"][1],
                "distractors": litqa2_test_data["distractors"][1]
            },
            {
                "question": litqa2_test_data["question"][2],
                "ideal": litqa2_test_data["ideal"][2],
                "distractors": litqa2_test_data["distractors"][2]
            }
        ]
    )
    
    dataset = df_2_sample(mini_data)
    
    MULTIPLE_CHOICE_TEMPLATE = """
    The following is a multiple choice question about biology.
    Please answer by responding with the letter of the correct answer.

    Think step by step.

    Question: {question}
    Options:
    {choices}

    You MUST include the letter of the correct answer within the following format: 'ANSWER: $LETTER' (without quotes). For example, ’ANSWER: <answer>’, where <answer> is the correct letter. Always answer in exactly this format of a single letter, even if you are unsure. We require this because we use automatic parsing.
    """
    
    UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."
    @task
    def defualt_llm_task() -> Task:
        return Task(
            dataset=dataset,
            solver=[multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE, cot=True)],
            scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
            epochs=Epochs(1, "mode")
            
        )
        
    eval_result = eval(
        tasks=defualt_llm_task,
        time_limit=300.0,
        max_samples=3,
        model="openai/gpt-4o-mini"
    )