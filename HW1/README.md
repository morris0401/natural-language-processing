# HW1

This directory contains the files for **HW1**.

## 📄 Project Introduction

### Project Introduction

This assignment focuses on the Word Analogy task, a method used to evaluate how effectively word embedding models capture semantic and syntactic relationships between words. The task is conceptually represented as "A is to B as C is to D" and mathematically as `vector(B) - vector(A) + vector(C) ≈ vector(D)`.

### Methods Used

*   **Dataset:** The Google Word Analogy dataset (Mikolov et al., 2013) is used, comprising 19,544 entries categorized into 5 semantic and 9 syntactic sub-categories. Each entry consists of four words (A B C D), where D is the target answer.
*   **Word Embeddings:**
    *   **Pre-trained:** GloVe-wiki-gigaword-100 embeddings are loaded using the Gensim library.
    *   **Custom-trained:** A Word2Vec model is trained using the Gensim library on a sampled Wikipedia corpus.
*   **Analogy Prediction:** Vector arithmetic is employed, where the predicted word `D` is the closest word to `vector(B) - vector(A) + vector(C)` in the embedding space, typically using cosine similarity.
*   **Visualization:** t-Distributed Stochastic Neighbor Embedding (t-SNE) is used to visualize word relationships, specifically for words within the 'family' sub-category.
*   **Training Corpus:** A pre-processed subset (20% sample) of the English Wikipedia dump is used for training custom word embeddings.
*   **Corpus Pre-processing (for custom training):** Recommended steps include removing non-English words, stop words, lemmatization (e.g., 'rocks' -> 'rock'), advanced tokenization, and retaining only the most frequent words.
*   **Evaluation Metrics:** Accuracy is calculated by comparing predicted words with gold answers, reported for overall performance, main categories (semantic/syntactic), and individual sub-categories.

### Experimental Procedures

The assignment involves a series of seven tasks, executed primarily in a Google Colab environment:

1.  **Data Pre-processing (Google Analogy Dataset):** Download the `questions-words.txt` file, then convert it into a pandas DataFrame, correctly classifying entries into semantic and syntactic categories.
2.  **Analogy Prediction (Pre-trained Embeddings):** Load a pre-trained GloVe model. For each analogy entry (A B C D), calculate `vector(B) - vector(A) + vector(C)` and find the nearest word `D_pred` in the embedding space. Store `D_pred` and the gold answer `D`.
3.  **Visualization (Pre-trained Embeddings):** Plot a t-SNE visualization of word relationships for the 'family' sub-category using the pre-trained GloVe embeddings.
4.  **Wikipedia Corpus Sampling:** Download pre-cleaned and split Wikipedia articles, combine them, and then sample 20% of the combined articles to create a smaller training corpus.
5.  **Word Embedding Training (Custom):** Apply recommended pre-processing steps (e.g., stop word removal, lemmatization, tokenization) to the sampled Wikipedia articles and then train a Gensim Word2Vec model on this processed corpus.
6.  **Analogy Prediction (Custom-trained Embeddings):** Repeat the analogy prediction task (similar to step 2) using the newly trained Word2Vec embeddings.
7.  **Visualization (Custom-trained Embeddings):** Plot a t-SNE visualization of word relationships for the 'family' sub-category using the custom-trained Word2Vec embeddings.

Evaluation of both pre-trained and custom-trained models is performed by comparing predicted words to gold answers, calculating accuracy across categories and sub-categories.

### Summary of Experimental Results (Example)

While explicit numerical results for the assignment are not provided in the document, an example t-SNE plot for the 'family' sub-category using pre-trained embeddings demonstrates the expected outcome. The plot visually groups words with similar semantic or syntactic relationships, such as 'man', 'woman', 'girl', 'boy', or 'king', 'queen', 'prince', 'princess', indicating that the embeddings capture these relationships. The objective is to compare how well pre-trained vs. custom-trained embeddings capture these relationships based on accuracy and t-SNE plots.

### Potential Improvements

*   **Model Exploration:** Experiment with different pre-trained word embedding models (e.g., FastText, different GloVe sizes) and explore various hyperparameters for the custom Word2Vec training (e.g., vector size, window size, min_count, training algorithm).
*   **Pre-processing Techniques:** Investigate the impact of different data pre-processing steps for both the analogy dataset and the Wikipedia corpus (e.g., alternative tokenizers, stemming vs. lemmatization, custom stop word lists).
*   **Training Data Size:** Analyze the effect of increasing or decreasing the sampling ratio of the Wikipedia corpus on the quality and performance of custom-trained word embeddings.
*   **Detailed Analysis:** Conduct in-depth analysis of accuracy differences across semantic and syntactic categories/sub-categories, identifying strengths and weaknesses of each embedding approach.
*   **t-SNE Interpretation:** Provide detailed interpretations of the t-SNE visualizations, highlighting specific clusters or outlier words and explaining their implications for word relationships.
*   **Error Analysis:** Examine specific examples where the models fail to predict the correct analogy, and hypothesize reasons for these failures.

## 📂 Files

- `NLP_HW1_NYCU_111550177.docx`
- `NLP_HW1_NYCU_111550177.py`
- `NTHU-NLP-HW1-word-emb.pdf`
- `requirements.txt.txt`

A detailed project description and summary can be found in the [main repository README](../README.md).

