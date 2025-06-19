import pandas as pd

from paperqa import Settings
from paperqa.settings import AgentSettings, AnswerSettings

from paperqa2_analysis.evaluate import MultipleChoiceEval
from paperqa2_analysis.agents.paperqa_agent import paperqa_agent

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
    
    # Set up LLM config (main LLM for reasoning, extract metadata, ...)
    llm_config_dict = {
        "model_list": [
            {
                "model_name": "gpt-4.1",
                "litellm_params": {
                    "model": "gpt-4.1",
                    "temperature": 0,
                    "max_tokens": 4096
                }
            }
        ],
        "rate_limit": {"gpt-4.1": "30000 per 1 minute"}
    }

    # Set up agent (answer search and selecting tools):
    agent_settings = AgentSettings(
        agent_llm="gpt-4.1",
        agent_llm_config={
            "rate_limit": "30000 per 1 minute"
        }
    )

    # Set up summary LLM config
    summary_config_dict = {
        "rate_limit": {"gpt-4.1": "30000 per 1 minute"}
    }

    # Set up answer format
    answer_settings = AnswerSettings(
        evidence_k=30,
        evidence_detailed_citations=False,
        evidence_retrieval=False,
        evidence_summary_length="around 30 words",
        evidence_skip_summary=False,
        answer_max_sources=5,
        max_answer_attempts=1,
        answer_length="1 letter"
    )

    # Set up the final settings object
    paperqa_settings = Settings(
        llm="gpt-4.1",
        llm_config=llm_config_dict,
        summary_llm="gpt-4.1",
        summary_llm_config=summary_config_dict,
        temperature=0,
        batch_size=1,
        verbosity=1,
        paper_directory="/root/paperQA2_analysis/data/LitQA_data/LitQA2_test_pdfs"
    )
    
     # Create the evaluation instance with full test dataset
    eval_instance = MultipleChoiceEval(
        data=mini_data,
        agent=paperqa_agent,
        template=None,  # Will use default template
        settings=paperqa_settings  # Pass settings as a kwarg
    )
    
    # Run the evaluation
    print("\nRunning evaluation on test dataset...")
    eval_instance.run(
        max_samples=3,
        time_limit=300.0
    )
    print("Evaluation complete!")
    