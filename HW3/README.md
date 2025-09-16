# HW3

This directory contains the files for **HW3**.

## 📄 Project Introduction

### Project Introduction
This assignment focuses on building a multi-output learning model for Natural Language Processing (NLP). The objective is to train a single model to predict multiple related labels from a given input text. The task utilizes the SemEval 2014 Task 1 dataset, where each data instance, consisting of a `premise` and a `hypothesis`, requires two simultaneous predictions: a `relatedness_score` (regression task, values 1-5) and an `entailment_judgement` (3-class classification: NEUTRAL, ENTAILMENT, CONTRADICTION).

### Methods Used
*   **Multi-output Learning Approach:** A single model processes the input (premise-hypothesis pair) to concurrently generate predictions for both semantic relatedness and textual entailment.
*   **Dataset:** SemEval 2014 Task 1, partitioned into 4500 training, 500 validation, and 4927 test samples. Each sample contains a premise, a hypothesis, a relatedness score, and an entailment judgment.
*   **Model Architecture:** A BERT encoder (from the Hugging Face API) serves as the core feature extractor. Its output is fed into two distinct linear layers: one for predicting the `relatedness_score` (Linear_1) and another for predicting the `entailment_judgement` (Linear_2).
*   **Optimizer:** Adam or AdamW is recommended for model optimization.
*   **Loss Functions:** Task-specific loss functions are applied, e.g., a regression loss (such as Mean Squared Error) for `relatedness_score` and a classification loss (such as Cross-Entropy) for `entailment_judgement`. The total loss is an aggregation of these individual task losses.
*   **Evaluation Metrics:**
    *   For `relatedness_score`: Spearman and Pearson correlation coefficients.
    *   For `entailment_judgement`: Accuracy and macro F1-score (for 3 classes). The `torchmetrics` package is recommended for computing these scores.

### Experimental Procedures
1.  **Data Loading and Preparation:** The SemEval 2014 Task 1 dataset is loaded using the `datasets` library (version 2.21.0) from Hugging Face. A custom PyTorch `collate_fn` is implemented to tokenize text inputs (premise and hypothesis) and organize batched data, ensuring separate handling of labels for the regression and classification sub-tasks.
2.  **Model Construction:** A `MultiLabelModel` (inheriting from `torch.nn.Module`) is implemented, incorporating a BERT model and two distinct output heads for the specified tasks.
3.  **Training Loop Implementation:** A custom PyTorch training loop is developed, foregoing the `HuggingFace Trainer`. This loop manages the forward pass, computes the aggregated loss, performs back-propagation, and executes optimizer steps for each batch.
4.  **Model Evaluation:** The model's performance is periodically evaluated on the validation set. During evaluation, Spearman/Pearson correlation coefficients are computed for relatedness, and Accuracy/macro F1-score are computed for entailment judgment. Model checkpoints are saved after each evaluation epoch.

### Summary of Experimental Results
The document outlines target performance metrics for the assignment but does not provide actual experimental results.
*   **Baseline Performance:** Students are expected to achieve a Spearman correlation coefficient of 0.71 and an Accuracy of 0.85 on the test set.
*   **Bonus Scores (using BERT-base only):**
    *   Spearman correlation coefficient > 0.74: 3% bonus.
    *   Spearman correlation coefficient > 0.77: 7% bonus.
    *   Spearman correlation coefficient > 0.8: 10% bonus.

### Potential Improvements
The assignment requires students to include a dedicated section in their report discussing strategies for improving model performance. This includes justifying the choice of pre-trained model, comparing the performance of multi-output learning against models trained separately for each sub-task, and conducting an error analysis to understand why the model fails on certain data points. These discussions contribute significantly to the project's overall grading.

### How to Run and Reproduce
Detailed, step-by-step instructions or runnable code for reproducing the project are not provided in this document. The PDF offers "hints only" for implementation, outlining the required structure for data loading, model definition, optimizer and loss function setup, and the training/evaluation loops. It specifies using `datasets==2.21.0` and requires reporting the running environment (e.g., Python version, GPU used). Students are expected to implement the project based on these conceptual guidelines.

## 📂 Files

- `NLP_HW3_NYCU_111550177.docx`
- `NLP_HW3_NYCU_111550177.py`
- `NTHU-NLP-HW3-Multi-output-learning.pdf`
- `requirements.txt`

A detailed project description and summary can be found in the [main repository README](../README.md).

