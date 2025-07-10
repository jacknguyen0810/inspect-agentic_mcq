# PaperQA2 Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13740-b31b1b.svg)](http://arxiv.org/abs/2409.13740)

This package extends the popular `inspect_ai` framework to provide flexible MCQ evaluation capabilities for agentic RAG systems.

## Overview

This project reproduces and extends the evaluation of PaperQA2's performance on the LitQA benchmark, as described in the paper ["Language agents achieve superhuman synthesis of scientific knowledge"](http://arxiv.org/abs/2409.13740). The framework provides tools for:

- **Reproducible Evaluation**: Systematic testing of PaperQA2 configurations
- **Multi-Agent Integration**: Custom wrapper systems for structured evaluation
- **Comprehensive Analysis**: Hyperparameter studies and performance comparisons
- **Standardized Metrics**: Accuracy, precision, recall, and F1-score calculations

## Key Features

- **Inspect AI Integration**: Easy integration into Inspect Ai for multi-agent evaluation. 
- **Custom PaperQA Agent**: For ease of use and integration with Inspect Ai. 

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (for GPT models, optional)
- Google API key (for Gemini models, optional)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/paperQA2_analysis.git
cd paperQA2_analysis

# Install in development mode
pip install -e .
```

### Install Dependencies

The package will automatically install required dependencies:

- `inspect-ai`: Evaluation framework
- `paper-qa>=5`: PaperQA2 implementation
- `ag2[openai]`: Agent framework
- `pydantic`: Data validation
- `pandas`: Data manipulation

## 🔧 Quick Start

### Basic Usage

```python
from paperqa2_analysis.agents.paperqa_agent import PaperQAAgent
from paperqa2_analysis.evaluate import evaluate_agent

# Initialize PaperQA agent
agent = PaperQAAgent(
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    max_sources=5,
    evidence_k=15
)

# Evaluate on test data
results = evaluate_agent(agent, test_data)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Precision: {results['precision']:.2%}")
```

### Advanced Configuration

```python
from paperqa2_analysis.agents.paperqa_gemini_embed_agent import PaperQAGeminiEmbedAgent

# Use Google's advanced embedding model
agent = PaperQAGeminiEmbedAgent(
    model="gpt-4o-mini",
    embedding_model="text-embedding-004",
    max_sources=10,
    evidence_k=20,
    use_rcs=True  # Enable re-ranking and contextual summarization
)
```

## 📊 Data Format

The framework expects data in the following format:

```python
import pandas as pd

# Required columns
data = pd.DataFrame({
    'question': ['What is the main finding of the study?', ...],
    'ideal': ['A', 'B', 'C', 'D'],  # Correct answers
    'distractors': [['B', 'C', 'D'], ['A', 'C', 'D'], ...]  # Incorrect options
})
```

## 🧪 Available Agents

### PaperQAAgent
Standard PaperQA2 implementation with OpenAI models.

```python
from paperqa2_analysis.agents.paperqa_agent import PaperQAAgent

agent = PaperQAAgent(
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small"
)
```

### PaperQAGeminiEmbedAgent
PaperQA2 with Google's advanced embedding models.

```python
from paperqa2_analysis.agents.paperqa_gemini_embed_agent import PaperQAGeminiEmbedAgent

agent = PaperQAGeminiEmbedAgent(
    model="gpt-4o-mini",
    embedding_model="text-embedding-004"
)
```

### BridgeAgent
Multi-agent wrapper for structured evaluation.

```python
from paperqa2_analysis.agents.bridge_agent import BridgeAgent

agent = BridgeAgent(
    primary_agent=paperqa_agent,
    parser_agent=parser_agent
)
```

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Accuracy**: Overall correctness across all questions
- **Precision**: Correctness of answered questions
- **Recall**: Coverage of questions attempted
- **F1-Score**: Harmonic mean of precision and recall
- **PaperQA Score**: Custom metric from original paper
- **Answered Recall**: Accuracy on attempted questions

## 🔬 Research Results

Our reproduction study achieved:

- **Superhuman Precision**: All RAG configurations achieved >77.3% precision (human benchmark: 73.8%)
- **Best Performance**: GPT-4o-Mini + text-embedding-004 achieved 89.5% precision
- **Robust Performance**: Near-superhuman results even with suboptimal hyperparameters
- **Reproducibility Challenges**: Identified performance fluctuations due to API load and hardware

## 📁 Project Structure

```
paperQA2_analysis/
├── paperqa2_analysis/          # Main package
│   ├── agents/                 # Agent implementations
│   ├── inspect_ai_custom/      # Custom evaluation components
│   └── evaluate.py            # Evaluation utilities
├── demo/                      # Example scripts and notebooks
├── data/                      # LitQA dataset
├── logs/                      # Evaluation logs
├── report/                    # Research paper
└── summary/                   # Executive summary
```

## 🎯 Examples

### Single Question Evaluation

```python
from demo.paperqa_single_demo import evaluate_single_question

result = evaluate_single_question(
    question="What is the main finding?",
    options=["A", "B", "C", "D"],
    correct_answer="A",
    agent=agent
)
```

### Full Benchmark Evaluation

```python
from demo.full_demo import run_full_evaluation

results = run_full_evaluation(
    test_data=test_df,
    agent_configs=configurations,
    num_runs=3
)
```

### Hyperparameter Study

```python
from demo.answer_cutoff import study_answer_cutoff

results = study_answer_cutoff(
    agent=agent,
    test_data=test_df,
    cutoff_values=[5, 10, 15]
)
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # Optional
```

### Model Parameters

```python
# Retrieval parameters
max_sources = 5      # Number of sources to include in answer
evidence_k = 15      # Number of evidence chunks to retrieve

# Search parameters
top_k = 30          # Number of chunks to retrieve initially
use_rcs = True      # Enable re-ranking and contextual summarization

# Model selection
model = "gpt-4o-mini"           # LLM for reasoning
embedding_model = "text-embedding-3-small"  # Embedding model
```

## 📊 Performance Comparison

| Configuration | Accuracy | Precision | Model |
|---------------|----------|-----------|-------|
| GPT-4o-Mini + text-embedding-004 | 82.1% | 89.5% | Best |
| GPT-4-Turbo + text-embedding-3-small | 78.9% | 85.2% | Baseline |
| GPT-4.1 + text-embedding-3-small | 71.2% | 77.3% | Subpar |
| Human Benchmark | 73.8% | 73.8% | Reference |

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limits**: Use sequential execution instead of parallel
2. **Timeout Errors**: Increase timeout settings or reduce batch size
3. **Memory Issues**: Reduce `max_sources` or `evidence_k` parameters
4. **Embedding Errors**: Verify API keys and model availability

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
agent = PaperQAAgent(debug=True)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black paperqa2_analysis/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{skarlinski_language_2024,
    title={Language agents achieve superhuman synthesis of scientific knowledge},
    author={Skarlinski, Michael D. and Cox, Sam and Laurent, Jon M. and Braza, James D. and Hinks, Michaela and Hammerling, Michael J. and Ponnapati, Manvitha and Rodriques, Samuel G. and White, Andrew D.},
    journal={arXiv preprint arXiv:2409.13740},
    year={2024}
}
```

## 📞 Contact

- **Author**: Phong-Anh Nguyen Trinh
- **Email**: pan31@cam.ac.uk
- **Institution**: Department of Physics, University of Cambridge

## 🙏 Acknowledgments

- Original PaperQA2 authors for the foundational work
- Inspect AI team for the evaluation framework
- OpenAI and Google for providing API access

---

**Note**: This is a research reproduction project. Results may vary depending on API availability, model updates, and system configurations.




