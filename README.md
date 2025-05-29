# paperQA2_analysis
This package gives an extension to the popular ```inspect_ai``` package, allowing more flexibility into the bridge function. 

The package is currently built for evaluating agents built for answering multiple choice questions.

## Installation
1. Clone from Github

2. Run the following command:

```
pip install -e .
```

## Data Format

Data must be in the form of a ```pandas``` DataFrame. It must contain the columns: question (question to be asked to the agent system), ideal (correct answer to the question), distractors (incorrect answers). 




