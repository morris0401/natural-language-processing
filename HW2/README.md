# HW2

This directory contains the files for **HW2**.

## 📄 Project Introduction

### Project Introduction
This assignment focuses on training sequence generation models to treat arithmetic expressions as a specialized language. The core task involves developing and training a recurrent neural network (RNN), specifically a Long Short-Term Memory (LSTM) model, to accurately predict the answer to given arithmetic equations. A key objective is to practice training and analyzing such a model, and to reflect on its logical understanding of arithmetic operations.

### Methods Used
*   **Model Architecture:** A character-level Recurrent Neural Network (CharRNN) is employed, built using `torch.nn.Module`. It consists of:
    *   An embedding layer (`torch.nn.Embedding`) to convert input characters into dense vector representations.
    *   One or two LSTM layers (`torch.nn.LSTM`) for sequence processing, configured with `batch_first=True`.
    *   A linear output layer (`torch.nn.Sequential` including `torch.nn.Linear` and `torch.nn.ReLU`) to map hidden states to vocabulary logits, predicting the next character.
*   **Training Paradigm:** Teacher Forcing is a mandatory training technique. During training, the true target sequence (ground truth) from the dataset is fed as the input for the next time step, instead of the model's own predicted output.
*   **Optimization & Hardware:** Standard deep learning training procedures are followed, involving defining an optimizer and a loss function (implied to be suitable for character-level prediction, e.g., Cross-Entropy Loss). Model parameters are updated via backpropagation. Training is explicitly required to be performed on a GPU for efficiency.
*   **Preprocessing:** A vocabulary (character-to-ID and ID-to-character mappings) is constructed, including special tokens `<pad>` for padding shorter sequences and `<eos>` to mark the end of a generated sequence. Input sequences for training are formed by concatenating the arithmetic expression with its ground truth answer. For loss calculation, only the tokens predicted *after* the `=` symbol are considered relevant, with preceding tokens in the target sequence replaced by `<pad>`.
*   **Inference (Generation):** A `generator` function is implemented to produce the answer character by character. The generation process starts with a `start_char` (typically the first character after `=`) and continues until the `<eos>` token is predicted or a maximum length is reached.
*   **Evaluation Metric:** The model's performance is measured using "Exact Match" accuracy on the evaluation set, meaning the entire generated answer must precisely match the ground truth.

### Experimental Procedures
1.  **Dataset Loading and Preparation:**
    *   Load the `arithmetic_train.csv` and `arithmetic_eval.csv` datasets, which contain 2-3 number arithmetic equations. Numbers are within the range [0, 50), and operations include `+`, `-`, `*`, `()`.
    *   The training set contains 2,369,250 pieces, and the evaluation set contains 263,250 pieces. Each entry includes an `input` expression (e.g., `(10 + 4) * 2 =`) and its `ground_truth` answer (e.g., `28`).
2.  **Vocabulary Construction:** Build a complete character dictionary from the training data, mapping all unique characters (digits, operators, `=`, `<pad>`, `<eos>`) to unique integer IDs.
3.  **Data Preprocessing and Batching:**
    *   Transform all input and target data into string format and then into numerical ID sequences using the created vocabulary.
    *   For loss calculation, create target sequences where tokens before the `=` symbol are masked (set to `<pad>`).
    *   Implement `torch.utils.data.Dataset` and `DataLoader` classes to efficiently handle data loading, padding, and batching.
4.  **Model Training:**
    *   Instantiate the `CharRNN` model.
    *   Train the model iteratively over epochs using the prepared training data, an optimizer (e.g., SGD), and a suitable loss function.
    *   Crucially, apply the teacher forcing technique during each training step.
    *   Ensure training is performed on a GPU.
5.  **Model Evaluation:**
    *   After training, use the `generator` function to produce predicted answers for all expressions in the evaluation set.
    *   Calculate the Exact Match accuracy by comparing these generated answers to their corresponding ground truths.

### Summary of Experimental Results
The provided PDF serves as an assignment description and framework, outlining the problem, methods, and procedures. It does not present specific experimental results or performance metrics for the arithmetic sequence generation task. It includes a general linear regression example (pages 4-8) to illustrate PyTorch concepts like loss reduction over epochs, but these results are not related to the arithmetic problem itself.

### Potential Improvements
The assignment's grading criteria suggest several avenues for further research and analysis that would enhance the project:
*   **Learning Rate Analysis:** Investigate how different learning rates impact the model's training dynamics, convergence speed, and final performance.
*   **Architectural Alternatives:** Compare the answer generation quality and efficiency when using other recurrent architectures (e.g., simple RNN or GRU) instead of LSTM.
*   **Generalization Capabilities:** Evaluate the model's ability to generalize to numbers outside the training distribution, such as training with 2-digit numbers and testing with 3-digit numbers.
*   **Robustness to Unseen Tokens:** Analyze the model's performance when the evaluation set contains numbers or operators that were not present in the training data.
*   **Gradient Clipping:** Study the necessity and effects of implementing gradient clipping during training to prevent exploding gradients and improve stability.
*   **Additional Analysis:** Incorporate any other insights or advanced analyses that contribute to a deeper understanding of the model's behavior or performance.

### How to Run and Reproduce
The assignment instructions provide a general workflow, primarily recommending Google Colab for execution.
*   **Environment Setup:**
    *   **Google Colab:** Connect to Google Drive to access and upload data files (`arithmetic_train.csv`, `arithmetic_eval.csv`, `main.ipynb`).
    *   **Local Setup:** For running locally, instructions for installing Miniconda are provided to set up a Python environment.
*   **Code Structure:** The project is expected to be developed within a Jupyter Notebook (`main.ipynb`), where students are required to fill in designated "TODO" sections for tasks such as dictionary building, data preprocessing, custom `Dataset` and `DataLoader` implementation, model definition, and generation logic.
*   **Data Files:** The `arithmetic_train.csv` and `arithmetic_eval.csv` datasets are crucial and need to be accessible to the Colab environment (e.g., via Google Drive).
*   **Submission:** The final submission package must include a Python script (`.py`), a `requirements.txt` file listing dependencies (e.g., `numpy==1.26.3`), and a Microsoft Word report (`.docx`). These files must follow specific naming conventions (e.g., `NLP_HW2_school_student_ID.py`) and be compressed into a single `.zip` archive.

## 📂 Files

- `NLP_HW2_NYCU_111550177.docx`
- `NLP_HW2_NYCU_111550177.py`
- `NTHU-NLP-HW2-Arithmetic-Updated.pdf`
- `requirements.txt`

A detailed project description and summary can be found in the [main repository README](../README.md).

