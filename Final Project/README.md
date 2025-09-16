# Final Project

This directory contains the files for **Final Project**.

## 📄 Project Introduction

# Bridging LLM Capabilities and Human Preferences: A DeBERTa-based Approach to User Preference Prediction

## Introduction

The rapid advancement of Large Language Models (LLMs) necessitates effective mechanisms to align their outputs with nuanced human preferences. This project addresses the critical task of predicting user preferences for LLM-generated responses, a challenge central to Reinforcement Learning from Human Feedback (RLHF) and vital for fostering more human-aligned and satisfactory AI interactions. Our objective is to reduce the disparity between LLM capabilities and human expectations by accurately modeling user choices.

We utilize a comprehensive dataset derived from user interactions on ChatBot Arena, comprising approximately 55,000 training and 25,000 test samples. Each entry includes a `prompt` and two LLM responses (`response_a`, `response_b`), alongside a `winner_model_[a/b/tie]` label indicating the user's preferred response or a tie. The task requires predicting three probabilities (preference for A, B, or tie) which sum to 1. Model performance is rigorously evaluated using Log Loss.

Key challenges in this domain include managing data heterogeneity arising from diverse LLMs and prompt styles, effectively processing lengthy texts without information loss, performing accurate multi-class classification, and ensuring robust generalization to unseen scenarios. Our exploratory data analysis revealed a rich landscape of 64 distinct LLMs, with `gpt-4-1106-preview`, `claude-2.1`, and `gpt-4-0613` being among the most frequently compared models, and ties accounting for a significant 31% of comparisons. This work contributes an efficient and accurate methodology for predicting user preferences, validated by competitive performance metrics.

## Methods

Our methodology focuses on leveraging established transformer architectures and fine-tuning techniques to develop a robust preference prediction system.

### Data Preprocessing
Initial steps involve comprehensive text preprocessing using NLTK for tokenization, removal of stopwords and special characters, and lemmatization to reduce noise and enhance semantic focus. The `winner_model_[a/b/tie]` labels are transformed into numerical targets, and `prompt`-`response` pairs are formatted for model input.

### Model Architecture
The core of our solution employs **DeBERTa-v3 small** as the base model. Selected for its efficiency and advanced contextual understanding, DeBERTa-v3 small processes sequences up to 512 tokens using its Disentangled Attention mechanism, which effectively separates content and position embeddings. The hidden layer outputs from DeBERTa-v3 are mean-pooled, concatenated, and then fed into a fully connected classification head. A Softmax activation function converts the final logits into a three-way probability distribution representing preference for response A, response B, or a tie.

### Training Configuration
Training is performed using `CrossEntropyLoss`, which directly aligns with the Log Loss evaluation metric. The `AdamW` optimizer is utilized, coupled with a linear learning rate scheduler for optimized training dynamics. To enhance training efficiency and prevent numerical instability, we integrate `torch.cuda.amp` for mixed-precision training and `amp.GradScaler` for gradient scaling during backpropagation.

### Exploratory Investigations
Beyond the primary DeBERTa model, we conducted supplementary investigations into alternative approaches:
*   **LoRA Fine-tuning on Llama-3.2-1B**: Explored efficient fine-tuning of a smaller LLM for enhanced emotional and semantic comprehension with reduced computational overhead.
*   **Direct Preference Optimization (DPO)**: Investigated DPO as a method to directly optimize user preferences from comparative data, aiming to simplify the learning process compared to traditional RLHF by removing the explicit reward model.

## Experimental Setup and Results

### Dataset
Our experiments utilized the ChatBot Arena dataset with 55,000 training entries and 25,000 test entries. Inputs included `prompt`, `response_a`, and `response_b`, with the target being a three-category preference label. Log Loss served as the primary evaluation metric.

### Main Model Training (DeBERTa-v3 small)
The DeBERTa-v3 small model was trained with `max_length = 512`, `epochs = 3`, `batch_size = 16`, and a `learning_rate = 2e-5`. A critical finding emerged regarding the application of the `softmax` function:
*   **Optimal Performance**: The lowest Log Loss of **1.04219** was achieved when raw logits from the model were fed directly into `CrossEntropyLoss` during training (i.e., `softmax` was *not* applied before loss calculation), and `softmax` was applied *only* during the inference stage to convert logits into probabilities.
*   **Impact of Softmax Placement**: Applying `softmax` before `CrossEntropyLoss` led to significantly higher Log Loss values (e.g., 1.04398 for 1 epoch, 1.07435 with `log_softmax`, and much higher when `softmax` was applied before loss *and not* during inference). This is attributed to two main issues: numerical instability from extreme logits causing overflow/underflow in exponential functions, and redundant computation as `CrossEntropyLoss` internally computes `log(softmax)`. This emphasizes the importance of feeding raw logits to `CrossEntropyLoss` for both numerical stability and computational efficiency.

### Exploratory Experiment Results
*   **System Prompt Experiment (LoRA Llama-3.2-1B)**: A generic system prompt provided only a marginal improvement for the LoRA-tuned Llama-3.2-1B model, reducing Log Loss from 1.7 to 1.65.
*   **Direct Preference Optimization (DPO)**: While conceptually explored for its theoretical advantages in directly optimizing user preferences, specific experimental results or performance metrics from its implementation were not detailed, indicating it was primarily an exploratory discussion.

## Future Work

Future directions aim to further enhance model performance, robustness, and interpretability:

*   **Enhanced Data Strategies**: Increase dataset diversity and implement feature-specific partitioning to mitigate overfitting. Employ data augmentation techniques, such as swapping `response_a` and `response_b` (along with their labels), to expand training data variations.
*   **Model Ensemble and Fusion**: Combine outputs from multiple LLMs (e.g., DeBERTa, Gemma, LLaMA) to leverage their individual strengths, thereby reducing the biases inherent in single-model approaches and potentially achieving higher accuracy.
*   **Resource and Overfitting Management**: Implement sparse fine-tuning methods like LoRA more broadly to selectively adjust a subset of model parameters, conserving computational resources and improving generalization.
*   **Advanced Feature Engineering and Model Selection**: Incorporate `TfidfVectorizer` and `CountVectorizer` for richer feature representation, combined with `LightGBM` for feature importance assessment and selection. Utilize embeddings of `prompt` and `response_a/b`, along with their attention scores, as additional features, potentially training models like `XGBoost` with early stopping.
*   **Robust Training Techniques**: Integrate Layer Normalization within the classifier for faster model convergence and apply Gradient Clipping during backpropagation to prevent exploding gradients.
*   **Multi-dimensional Evaluation Framework**: Introduce a broader spectrum of evaluation metrics beyond Log Loss, such as language fluency, semantic coherence, and explicit user satisfaction scores, for a more holistic assessment of human preference alignment.
*   **Scalability**: Explore the use of larger and more powerful backbone LLMs and expand the model ensemble to include a broader range of diverse models, given increased computational resources.

## 📂 Files

- `2024-NLP-term-project.pdf`
- `LLM Classification Finetuning.pdf`
- `NLP_term_project_report.pdf`
- `lmsys-kerasnlp-starter.ipynb`
- `model.ipynb`
- `post_template_group87.pdf`

A detailed project description and summary can be found in the [main repository README](../README.md).

