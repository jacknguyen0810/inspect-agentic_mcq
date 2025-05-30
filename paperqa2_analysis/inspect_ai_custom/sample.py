import random

from pandas import DataFrame

from inspect_ai.dataset import MemoryDataset, Sample


def record_to_sample_custom(record: dict) -> Sample:
    """Custom function to transform dictionaries into inspect_ai Samples.

    Args:
        record (dict): Conatins information needed for MCQ.

    Returns:
        Sample: Completed Sample object for MCQ
    """
    # Get the question
    message = f"Question: {record["question"]} \n"

    # Concatenate the choices
    choices = [record["ideal"]]
    choices.extend(record["distractors"])

    # Shuffle because we want the final answer to be unsure
    # Shuffle the dataset
    random.shuffle(choices)

    # Find the ideal answer
    ideal_idx = choices.index(record["ideal"])

    # Add prefixes to the shuffled choices
    # indices = list[range(len(choices))]
    message += "\n".join([f"{chr(65 + i)}) {j}" for i, j in enumerate(choices)])

    message += f"\nNA) {UNCERTAIN_ANSWER_CHOICE}"

    # Add the target to the message:
    message += f"\n\nTarget: {chr(65 + ideal_idx)}"

    # Make the message a part of the Sample
    return Sample(input=message, choices=choices, target=f"{chr(65 + ideal_idx)}")


def df_2_sample_bridge(data: DataFrame) -> MemoryDataset:
    """Function to transform a pandas DataFrame to a MemoryDataset for inspect_ai processing.

    Args:
        data (DataFrame): DataFrame containing required information.

    Returns:
        MemoryDataset: Full MemoryDataset for inspect_ai processing
    """
    records = data.to_dict(orient="records")
    samples = [record_to_sample_custom(i) for i in records]
    return MemoryDataset(samples)


UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."

def record_to_sample(record: dict) -> Sample:

    # Concatenate the choices
    choices = [record["ideal"]]
    choices.extend(record["distractors"])

    # Shuffle because we want the final answer to be unsure
    # Shuffle the dataset
    random.shuffle(choices)

    # Find the ideal answer
    ideal_idx = choices.index(record["ideal"])

    # Make the message a part of the Sample
    return Sample(input=record["question"], choices=choices, target=f"{chr(65 + ideal_idx)}")

def df_2_sample(data: DataFrame) -> MemoryDataset:
    records = data.to_dict(orient="records")
    samples = [record_to_sample(i) for i in records]
    return MemoryDataset(samples)


if __name__ == "__main__":
    import pandas as pd

    # Import data
    litqa2_test_data = pd.read_parquet(
        "/root/paperQA2_analysis/data/LitQA_data/test-00000-of-00001.parquet"
    )

    # Create a test dataset
    test_dataset = df_2_sample_bridge(litqa2_test_data)
    print("-" * 20)
    print(test_dataset.samples[0])
    print(test_dataset.samples[1])
    print("-" * 20)

    example = {
        "question": litqa2_test_data["question"][0],
        "ideal": litqa2_test_data["ideal"][0],
        "distractors": litqa2_test_data["distractors"][0],
    }

    sample = record_to_sample_custom(example)
    mini_dataset = MemoryDataset([sample])

    print(mini_dataset.samples[0].input)
