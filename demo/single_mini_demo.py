import pandas as pd

from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.agent import bridge


from paperqa2_analysis.inspect_ai_custom.sample import record_to_sample_custom, df_2_sample_bridge, UNCERTAIN_ANSWER_CHOICE
from paperqa2_analysis.inspect_ai_custom.paperqa_scorer import paperqa_scorer
from paperqa2_analysis.agents.paperqa_agent import paperqa_agent

if __name__ == "__main__":
    # Import data
    litqa2_test_data = pd.read_parquet("/root/paperQA2_analysis/data/LitQA_data/test-00000-of-00001.parquet")
    
    # Mini Example with 1 working sample
    mini_example = {
        "question": litqa2_test_data["question"][0],
        "ideal": litqa2_test_data["ideal"][0],
        "distractors": litqa2_test_data["distractors"][0]
    }

    sample = record_to_sample_custom(mini_example)
    mini_dataset = MemoryDataset([sample])
    
    # Custom Task
    @task
    def paperqa_eval_mini():
        return Task(
            dataset=mini_dataset,
            solver = bridge(paperqa_agent()),
            scorer=paperqa_scorer(no_answer=UNCERTAIN_ANSWER_CHOICE),
            epochs=Epochs(1, "mode")
        )
        
    eval(paperqa_eval_mini())
    