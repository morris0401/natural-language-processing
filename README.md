# natural-language-processing

![language](https://img.shields.io/badge/language-Python-blue.svg)

Course materials for **natural language processing**.

## 📂 Course Assignments

- **[Final Project](Final Project)**: The rapid advancement of Large Language Models (LLMs) necessitates effective mechanisms to align their outputs with nuanced human preferences. This project addresses the critical task of predicting user preferences for LLM-generated responses, a challenge central to Reinforcement Learning from Human Feedback (RLHF) and vital for fostering more human-aligned and satisfactory AI interactions. Our objective is to reduce the disparity between LLM capabilities and human expectations by accurately modeling user choices.
- **[HW1](HW1)**: This assignment focuses on the Word Analogy task, a method used to evaluate how effectively word embedding models capture semantic and syntactic relationships between words. The task is conceptually represented as "A is to B as C is to D" and mathematically as `vector(B) - vector(A) + vector(C) ≈ vector(D)`.
- **[HW2](HW2)**: This assignment focuses on training sequence generation models to treat arithmetic expressions as a specialized language. The core task involves developing and training a recurrent neural network (RNN), specifically a Long Short-Term Memory (LSTM) model, to accurately predict the answer to given arithmetic equations. A key objective is to practice training and analyzing such a model, and to reflect on its logical understanding of arithmetic operations.
- **[HW3](HW3)**: This assignment focuses on building a multi-output learning model for Natural Language Processing (NLP). The objective is to train a single model to predict multiple related labels from a given input text. The task utilizes the SemEval 2014 Task 1 dataset, where each data instance, consisting of a `premise` and a `hypothesis`, requires two simultaneous predictions: a `relatedness_score` (regression task, values 1-5) and an `entailment_judgement` (3-class classification: NEUTRAL, ENTAILMENT, CONTRADICTION).
- **[HW4](HW4)**: This assignment focuses on implementing a Retrieval-Augmented Generation (RAG) system using LangChain for a Question Answering (QA) task. The goal is to answer ten specific questions about cats' knowledge by retrieving information from a provided text database. Answers are evaluated based on exact matching. A frozen Llama-3.2-1B model is used as the generator, with other components chosen and configured by the implementer.
