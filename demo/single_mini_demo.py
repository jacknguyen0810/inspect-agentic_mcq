import pandas as pd

from paperqa import Settings
from paperqa.settings import AgentSettings, AnswerSettings

from paperqa2_analysis.evaluate import MultipleChoiceEval
from paperqa2_analysis.agents.paperqa_agent import paperqa_agent

if __name__ == "__main__":
    # Import data
    litqa2_test_data = pd.read_parquet("/root/paperQA2_analysis/data/LitQA_data/test-00000-of-00001.parquet")
    
    # Create mini dataset with one sample
    mini_data = pd.DataFrame([{
        "question": litqa2_test_data["question"][0],
        "ideal": litqa2_test_data["ideal"][0],
        "distractors": litqa2_test_data["distractors"][0]
    }])
    
    # Set up LLM config (main LLM for reasoning, extract metadata, ...)
    llm_config_dict = {
        "model_list": [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "max_tokens": 4096
                }
            }
        ],
        "rate_limit": {"gpt-4o-mini": "30000 per 1 minute"}
    }

    # Set up agent (answer search and selecting tools):
    agent_settings = AgentSettings(
        agent_llm="gpt-4o-mini",
        agent_llm_config={
            "rate_limit": "30000 per 1 minute"
        }
    )

    # Set up summary LLM config
    summary_config_dict = {
        "rate_limit": {"gpt-4o-mini": "30000 per 1 minute"}
    }

    # Set up answer format
    answer_settings = AnswerSettings(
        evidence_k=30,
        evidence_detailed_citations=False,
        evidence_retrieval=False,
        evidence_summary_length="around 100 words",
        evidence_skip_summary=False,
        answer_max_sources=5,
        max_answer_attempts=5,
        answer_length="1 letter"
    )

    # Set up the final settings object
    paperqa_settings = Settings(
        llm="gpt-4o-mini",
        llm_config=llm_config_dict,
        summary_llm="gpt-4o-mini",
        summary_llm_config=summary_config_dict,
        agent=agent_settings,
        temperature=0,
        batch_size=1,
        verbosity=1,
        paper_directory="/root/paperQA2_analysis/data/LitQA_data/LitQA2_test_pdfs"
    )
    
    # Create the evaluation instance
    eval_instance = MultipleChoiceEval(
        data=mini_data,
        agent=paperqa_agent,
        template=None,  # Will use default template
        no_answer="NA"  # Specify the no answer option
    )
    
    # Run the evaluation
    print("\nRunning evaluation on mini dataset...")
    eval_instance.run()
    print("Evaluation complete!")
    