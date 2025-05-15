import os
import json

from autogen import ConversableAgent, LLMConfig
from pydantic import BaseModel, Field

# Using a Pydantic Base Class to structure the output of the agent
class StructuredInput(BaseModel):
    question: str = Field(..., description="Question, The question and multiple choice answers")
    target: str = Field(..., description="Target, just the letter of the target")
    
    def format(self) -> dict:
        return {"Question": self.question, "Target": self.target}

class StructuredOutput(BaseModel):
    answer: str = Field(..., description="Answer, the single letter answer to the question, in the format of LETTER or NA if NA is chosen")
    explanation: str = Field(..., description="Explanation, a short explanation of the answer with any citations found within the text.")
    citations: list[str] = Field(..., description="Citations, a list of citations found within the text.")
    target: str = Field(..., description="The target answer, in the format of LETTER")
    
    def format(self) -> dict:
        return {"Answer": self.answer, "Explanation": self.explanation, "Citations": self.citations, "Target": self.target}
        # return f"Answer: {self.answer}\nExplanation: {self.explanation}\nCitations: {self.citations}"
    
    
# Templates
AGENT_INSTRUCTIONS = """
You are an agent that is able to parse the output of a given text and return the desired output.
"""
ANSWER_MESSAGE_TEMPLATE = """
Please could you parse the following text and return the desired output.

Text:
{text}
"""

    
def structured_agent(
    input_text: str,
    structure: StructuredInput | StructuredOutput,
    model: tuple | None = None,
    temp: float = 0.1,
):
    # Default model to OpenAI gpt-4o-mini
    if model is None:
        model = ("openai", "gpt-4o-mini")
    
    llm_config = LLMConfig(
        api_type=model[0],
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model[1],
        temperature=temp,
        response_format=structure,
    )
    
    # Create agent 
    agent = ConversableAgent(
        name="structured_agent",
        llm_config=llm_config,
        system_message=AGENT_INSTRUCTIONS,
    )
    
    response = agent.run(
        message=ANSWER_MESSAGE_TEMPLATE.format(text=input_text),
        max_turns=1,
    )
    
    response.process()
    
    # Get the final message
    final_message = response.messages[-1]
    
    return final_message["content"]


if __name__ == "__main__":

    # Test
    import nest_asyncio
    nest_asyncio.apply()
    
    test_input = """Question: Approximately what percentage of topologically associated domains in the GM12878 blood cell line does DiffDomain classify as reorganized in the K562 cell line? 
    A) 31%
    B) 41%
    C) 11%
    D) 51%
    E) 21%
    F) Insufficient information to answer the question.
    Target: A
    """

    test_output = """Answer: DiffDomain identifies that approximately 30.771% of topologically associated domains (TADs) in  
            the GM12878 blood cell line are reorganized in the K562 cell line                                       
            (hua2024diffdomainenablesidentification pages 4-4). This finding is significant when compared to other  
            methods, such as TADCompare, HiCcompare, and HiC-DC+, which only identify ≤8.256% of GM12878 TADs as    
            reorganized in K562. The benchmarking results highlight the efficacy of DiffDomain in detecting         
            reorganized TADs between these cell lines (hua2024diffdomainenablesidentification pages 4-4).           
                                                                                                                    
            Additionally, the analysis indicates that the majority of identified reorganized TADs have a minimum of 
            43.137%, a median of 81.357%, and a maximum of 98.022% represented by other subtypes                    
            (hua2024diffdomainenablesidentification pages 4-5). This suggests a robust capability of DiffDomain in  
            identifying reorganized TADs, establishing a notable extent of reorganization between GM12878 and K562  
            (hua2024diffdomainenablesidentification pages 4-5).                                                     
                                                                                                                    
            In summary, the percentage of TADs in GM12878 classified as reorganized in K562 by DiffDomain is        
            approximately 30.771%, which aligns with option E in the multiple-choice question.                      
                                                                                                                    
            ANSWER: E
            Target: E"""
            
    na_test_prompt = """Not enough answer to answer the question. The selected answer is NA. 
    """

    struct_input = structured_agent(
        input_text=test_input,
        structure=StructuredInput
    )
    
    print(struct_input)
    print("-" * 50)
    
    struct_output = structured_agent(
        input_text=na_test_prompt,
        structure=StructuredOutput
    )
    
    print(struct_output)


    

