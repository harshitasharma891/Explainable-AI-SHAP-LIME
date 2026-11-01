# Explainable AI with SHAP and LIME

## Project Overview

This project demonstrates explainable AI techniques for understanding predictions made by a transformer-based text classification model.

The system combines semantic retrieval with SHAP and LIME to analyse how different words contribute to model predictions. A pre-trained DistilBERT sentiment classification model is used to generate predictions, while SHAP and LIME provide complementary explanations of the model's behaviour.

## Key Features

- Transformer-based text classification using DistilBERT.
- Semantic document retrieval using sentence embeddings.
- SHAP-based text explanations.
- LIME-based local explanations.
- Word-level feature importance analysis.
- Prediction probability analysis for positive and negative classes.
- Comparison of global and local explainability approaches.

## How the Project Works

The project follows this process:

```text
Input Text
    ↓
Text Classification with DistilBERT
    ↓
Prediction Probabilities
    ↓
SHAP Explanation ─────→ Word-Level Contributions
    ↓
LIME Explanation ─────→ Local Feature Importance
```

The notebook also includes semantic retrieval using sentence embeddings. Relevant documents are retrieved based on the input query and can be used to provide additional context.

## Models Used

### Text Classification Model

`distilbert-base-uncased-finetuned-sst-2-english`

Used for text classification and prediction probability analysis.

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

Used to generate embeddings and perform semantic similarity-based retrieval.

## Explainability Techniques

### SHAP

SHAP is used to analyse how input tokens contribute to the classification model's output.

The project uses:

- A tokenizer-based text masker.
- A custom model probability function.
- SHAP text explanations.

### LIME

LIME is used to generate local explanations for individual predictions.

The project identifies important words that influence the prediction probabilities for the classification classes.

## Technologies Used

- Python
- Hugging Face Transformers
- Sentence Transformers
- SHAP
- LIME
- PyTorch
- NumPy
- Google Colab

## Installation

Install the required libraries:

```bash
pip install transformers datasets shap lime faiss-cpu sentence-transformers torch scikit-learn
```

## Running the Project

1. Install the required libraries.
2. Load the DistilBERT text classification model.
3. Create the semantic retrieval component.
4. Define the model probability function.
5. Initialize the SHAP text explainer.
6. Generate a SHAP explanation for an input text.
7. Initialize the LIME text explainer.
8. Generate a LIME explanation and examine important words.

## Repository Structure

```text
Explainable-AI-SHAP-LIME/
│
├── SHAP_LIME.ipynb
└── README.md
```

## Author

**Harshita Sharma**

AI/ML Engineer

GitHub: github.com/harshitasharma891  
LinkedIn: linkedin.com/in/harshita-sharma-b843031b5
