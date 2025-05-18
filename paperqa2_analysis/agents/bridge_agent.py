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
):
    # Validate the custom agent before proceeding
    _validate_custom_agent(custom_agent)
    
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
        
        # Pass arguments to custom agent
        response = await custom_agent(query)
        
        # Add the target to the string response so that it can be parsed by the structured agent
        response += f"\nTarget: {target}"
        formatted = structured_agent(
            response,
            StructuredOutput
        )
        
        return dict(output=formatted)
    
    return run
        
        
        




def _validate_custom_agent(custom_agent: Callable) -> None:
    """Validate that the custom agent is a function that takes in and returns a string.
    
    Args:
        custom_agent: The custom agent function to validate
        
    Raises:
        TypeError: If the custom agent is not a function or doesn't have the correct signature
        ValueError: If the custom agent doesn't return a string
    """
    if not callable(custom_agent):
        raise TypeError("Custom agent must be a callable function")
        
    # Test the function with a sample input
    try:
        result = custom_agent("test question")
        if not isinstance(result, str):
            raise ValueError(f"Custom agent must return a string, got {type(result)}")
    except Exception as e:
        raise TypeError(f"Custom agent must accept a string input and return a string. Error: {str(e)}")   
        
        
        
MULTIPLE_CHOICE_TEMPLATE_BRIDGE = """
The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Think step by step.

{question}
"""

# Return your answer in the following format:

# "letter".

# where the letter denotes your chosen answer from the available options. You MUST only include the letter (with no quotation marks) and NOTHING ELSE.