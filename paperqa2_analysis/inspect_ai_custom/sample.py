import random

from pandas import DataFrame

from inspect_ai.dataset import MemoryDataset, Sample



def record_to_sample_custom(record: dict) -> Sample:
    # Get the question
    message = f"Question: {record["question"]} \n"
    
    # Concatenate the choices
    choices = [record["ideal"]]
    choices.extend(record["distractors"])
    
    # Shuffle because we want the final answer to be unsure
    # Shuffle the dataset
    random.shuffle(choices)
    
    
    choices.append(UNCERTAIN_ANSWER_CHOICE)
    
    # Find the ideal answer
    ideal_idx = choices.index(record["ideal"])
    
    # Add prefixes to the shuffled choices
    indices = list[range(len(choices))]
    message +=  "\n".join(
        [f"{chr(65 + i)}) {j}" for i, j in enumerate(choices)]
    )
    
    # Add the target to the message: 
    message += f"\nTarget: {chr(65 + ideal_idx)}"
    
    # Make the message a part of the Sample
    return Sample(
        input=message,
        choices=choices,
        target=f"{chr(65 + ideal_idx)}"
    )
    
    


def df_2_sample_bridge(data: DataFrame) -> MemoryDataset:
    records = data.to_dict(orient="records")
    samples = [record_to_sample_custom(i) for i in records]
    return MemoryDataset(samples)
    
    

UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."


if __name__ == "__main__":
    import pandas as pd
    
    # Import data
    litqa2_test_data = pd.read_parquet("/root/paperQA2_analysis/data/LitQA_data/test-00000-of-00001.parquet")
    
    # Create a test dataset
    test_dataset = df_2_sample_bridge(litqa2_test_data)
    print("-" * 20)
    print(test_dataset.samples[0])
    print(test_dataset.samples[1])
    print("-" * 20)
    
    example = {
        "question": litqa2_test_data["question"][0],
        "ideal": litqa2_test_data["ideal"][0],
        "distractors": litqa2_test_data["distractors"][0]
    }

    sample = record_to_sample_custom(example)
    mini_dataset = MemoryDataset([sample])
    
    print(mini_dataset.samples[0].input)