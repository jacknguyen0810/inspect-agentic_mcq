from collections.abc import Callable
import json

from inspect_ai.agent import agent

from paperqa2_analysis.agents.structured_agent import (
    structured_agent,
    StructuredInput,
    StructuredOutput,
)


@agent
def bridge_agent(custom_agent: Callable, template: str | None = None, **kwargs):
    """Custom agent wrapper to handle the bridging mechanic in inspect_ai. Deals with lack of options in TaskState by using AG2 agents to structure outputs into json schemas.

    Args:
        custom_agent (Callable): Function containing user's custom agent. E.g. def custom_agent(prompt: str, **kwargs)
        template (str | None, optional): Template for the prompt into custom agent. Must be able to format with a variable called 'question'. Defaults to None.
        **kwargs: Any kwargs needed for custom agent.

    Returns:
        Callable: Returns the run function for async processing.
    """

    # Give default template
    if template is None:
        template = MULTIPLE_CHOICE_TEMPLATE_BRIDGE

    async def run(sample: dict[str]) -> dict:
        print(sample)
        # Use structured agent to format the input
        input_result = structured_agent(sample["messages"][0]["content"], StructuredInput)
        message = json.loads(input_result["output"])
        question = message["question"]
        target = message["target"]

        # Format the template to pass to the agent
        query = template.format(question=question)

        # Pass arguments to custom agent, including any kwargs
        agent_result = await custom_agent(query, **kwargs)
        
        # Add the target to the string response so that it can be parsed by the structured agent
        # output_str = agent_result["answer"] + f"\nTarget: {target}"
        output_str = agent_result["answer"]
        formatted_result = structured_agent(output_str, StructuredOutput)
        
        # Pass the target after, avoid interaction with Structured Input
        output_dict = json.loads(formatted_result["output"])
        output_dict["Target"] = target
        output_json = json.dumps(output_dict)

        # Create the output dictionary with all metrics
        output = {
            "output": output_json,
            "cost": agent_result.get("cost", 0.0),
            "token_counts": agent_result.get("token_counts", {}),
            "metrics": {
                "cost": agent_result.get("cost", 0.0),
                "token_counts": agent_result.get("token_counts", {})
            }
        }

        return output

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

        test_sample = {"messages": [{"content": test_prompt}]}

        # Create the bridge agent with paperqa_agent as the custom agent
        # Pass paperqa_settings as a kwarg to be used by the custom agent
        test_agent = bridge_agent(
            custom_agent=paperqa_agent,
            template=MULTIPLE_CHOICE_TEMPLATE_BRIDGE,
            settings=paperqa_settings,  # This will be passed to paperqa_agent
        )

        # Run the agent
        result = await test_agent(test_sample)

        # Print the results
        print("\nTest Results:")
        print("-" * 50)
        print(f"Input question: {test_prompt.strip()}")
        print("\n")
        print(f"Bridge agent output: {result['output']}")
        print("-" * 50)

        # Verify the output format
        if not isinstance(result, dict):
            print("Error: Result is not a dictionary")
            return False

        if "output" not in result:
            print("Error: Result does not contain 'output' key")
            return False

        if not isinstance(result["output"], str):
            print("Error: Output is not a string")
            return False

        print("All tests passed!")
        return True

    # Run the test
    asyncio.run(test_bridge_agent())
