from paperqa import Settings, agent_query
from paperqa.settings import AgentSettings, AnswerSettings
            
        
async def paperqa_agent(
    prompt: str,
    settings: Settings | None = None
):  
    # Use provided settings or default to paperqa_settings
    settings_to_use = settings if settings is not None else paperqa_settings
    
    response = await agent_query(
        query=prompt,
        settings=settings_to_use
    )
        
    return response.session.answer


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


if __name__ == "__main__":
    # import os
    import asyncio
    
    # TODO: Update the test run
    
    test_prompt = """
    Question: Approximately what percentage of topologically associated domains in the GM12878 blood cell line does DiffDomain classify as reorganized in the K562 cell line? 
    A) 11%
    B) 41%
    C) 21%
    D) 51%
    E) 31%
    NA) Insufficient information to answer the question.
    Target: E
    """
    
    test_sample = {
        "messages": [
            {"content": test_prompt}
        ]
    }
    
    async def test_paperqa_agent():
        # Run the agent directly
        result = await paperqa_agent(
            prompt=test_prompt
        )
        
        # Print the result
        print("\nTest Results:")
        print("-" * 50)
        print(f"Input question: {test_prompt.strip()}")
        print("\n")
        print(f"Agent output: {result}")
        print("-" * 50)
        
        # Verify the output format
        if not isinstance(result, str):
            print("❌ Error: Result is not a string")
            return False
            
        print("✅ All tests passed!")
        return True
    
    # Run the test
    asyncio.run(test_paperqa_agent())
    