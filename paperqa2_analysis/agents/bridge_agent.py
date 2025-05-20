from collections.abc import Callable
import json
from typing import Any

from inspect_ai.agent import agent

from paperqa2_analysis.agents.structured_agent import (
    structured_agent,
    StructuredInput,
    StructuredOutput
)


@agent
def bridge_agent(
    custom_agent: Callable,
    template: str | None = None,
    **kwargs
):

    
    # Give default template
    if template is None:
        template = MULTIPLE_CHOICE_TEMPLATE_BRIDGE
    
    async def run(sample: dict[str]) -> dict:
        
        # Use structured agent to format the input
        input_str = structured_agent(sample["messages"][0]["content"], StructuredInput)
        message = json.loads(input_str)
        question = message["question"]
        target = message["target"]
        
        # Format the template to pass to the agent
        query = template.format(question=question)
        
        # Pass arguments to custom agent, including any kwargs
        response = await custom_agent(query, **kwargs)
        
        # Add the target to the string response so that it can be parsed by the structured agent
        response += f"\nTarget: {target}"
        formatted = structured_agent(
            response,
            StructuredOutput
        )
        
        return dict(output=formatted)
    
    return run 
        
        
        
MULTIPLE_CHOICE_TEMPLATE_BRIDGE = """
The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Think step by step.

{question}
"""

# Return your answer in the following format:

# "letter".

# where the letter denotes your chosen answer from the available options. You MUST only include the letter (with no quotation marks) and NOTHING ELSE.

if __name__ == "__main__":
    import asyncio
    from paperqa2_analysis.agents.paperqa_agent import paperqa_agent, paperqa_settings
    
    async def test_bridge_agent():
        # Create test input
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
        
        # Create the bridge agent with paperqa_agent as the custom agent
        # Pass paperqa_settings as a kwarg to be used by the custom agent
        agent = bridge_agent(
            custom_agent=paperqa_agent,
            template=MULTIPLE_CHOICE_TEMPLATE_BRIDGE,
            settings=paperqa_settings  # This will be passed to paperqa_agent
        )
        
        # Run the agent
        result = await agent(test_sample)
        
        # Print the results
        print("\nTest Results:")
        print("-" * 50)
        print(f"Input question: {test_prompt.strip()}")
        print("\n")
        print(f"Bridge agent output: {result['output']}")
        print("-" * 50)
        
        # Verify the output format
        if not isinstance(result, dict):
            print("❌ Error: Result is not a dictionary")
            return False
            
        if 'output' not in result:
            print("❌ Error: Result does not contain 'output' key")
            return False
            
        if not isinstance(result['output'], str):
            print("❌ Error: Output is not a string")
            return False
            
        print("✅ All tests passed!")
        return True
    
    # Run the test
    asyncio.run(test_bridge_agent())