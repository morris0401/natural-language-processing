# HW4

This directory contains the files for **HW4**.

## 📄 Project Introduction

### Project Introduction

This assignment focuses on implementing a Retrieval-Augmented Generation (RAG) system using LangChain for a Question Answering (QA) task. The goal is to answer ten specific questions about cats' knowledge by retrieving information from a provided text database. Answers are evaluated based on exact matching. A frozen Llama-3.2-1B model is used as the generator, with other components chosen and configured by the implementer.

### Methods Used

*   **Retrieval-Augmented Generation (RAG) Architecture:** The system combines a retriever and a frozen Large Language Model (LLM) generator to answer questions.
*   **Generator (LLM):** Llama-3.2-1B, run locally via Ollama, is used as the core generative model. The model is frozen, meaning its parameters are not fine-tuned.
*   **Retriever and Embedding Model:** `jinaai/jina-embeddings-v2-base-en` is employed through `HuggingFaceEmbeddings` to convert text documents into numerical vector representations.
*   **Vector Store:** Chroma is utilized as the vector database to store the embeddings of the cat facts and facilitate efficient retrieval of relevant documents.
*   **Framework:** LangChain is used to orchestrate the entire RAG pipeline, including document loading, embedding generation, vector store interaction, prompt engineering, and chaining the retrieval and generation components.
*   **Prompt Engineering:** A `system_prompt` is designed to guide the LLM on how to utilize the retrieved context to generate concise and accurate answers. `ChatPromptTemplate` is used to structure the input to the LLM.

### Experimental Procedures

1.  **Environment Setup:**
    *   Connect to Hugging Face by creating and logging in with an access token.
    *   Configure Google Colab to use a GPU runtime (e.g., T4 GPU).
    *   Install `colab-xterm` and Ollama.
    *   Pull the `llama3.2:1b` model using `ollama pull` and start the Ollama service (`ollama serve`) within the Colab terminal.
2.  **Model and Embedding Initialization:**
    *   Initialize the Ollama LLM instance with the `llama3.2:1b` model.
    *   Define the `HuggingFaceEmbeddings` model using `jinaai/jina-embeddings-v2-base-en`.
3.  **Dataset Loading and Retrieval Database Preparation:**
    *   Download the `cat-facts.txt` dataset from Hugging Face.
    *   Convert each fact into a `langchain_core.documents.Document` object, assigning a unique ID to each.
    *   Create a `Chroma` vector store from these documents using the defined embedding model.
    *   Configure a retriever from the vector store, specifying the `search_type` (e.g., "similarity", "mmr", or "similarity_score_threshold").
4.  **Prompt Configuration:**
    *   Formulate a `system_prompt` that instructs the LLM on its role and how to use the provided context to answer questions.
    *   Construct a `ChatPromptTemplate` incorporating the `system_prompt` and a placeholder for the human input.
5.  **RAG System Construction:**
    *   Build a `stuff_documents_chain` responsible for passing retrieved documents to the LLM for question answering.
    *   Integrate the retriever and the QA chain into a complete `retrieval_chain`.
6.  **Question Answering and Evaluation:**
    *   A set of ten predefined cat-related questions are iterated through.
    *   For each question, the `retrieval_chain` is invoked to obtain an answer.
    *   The generated answer is evaluated against a list of ground truth answers using an exact matching (case-insensitive) criterion. The number of correct answers is tracked.

### Summary of Experimental Results

The provided PDF outlines the assignment structure and implementation steps but does not include actual experimental results. The core task involves successfully building and running the RAG system to correctly answer ten questions about cat facts. The success metric is the count of questions for which the generated answer exactly matches the expected answer. The assignment requires students to perform these experiments and report their findings.

### Potential Improvements

*   **Advanced Prompt Engineering:** Experiment with different `system_prompt` variations, including detailed instructions, few-shot examples, or negative constraints, to enhance the LLM's ability to use context effectively and generate precise answers.
*   **Retriever Optimization:**
    *   **Search Type Comparison:** Analyze the performance impact of different `search_type` parameters in the retriever (e.g., "similarity", "mmr", "similarity_score_threshold") on retrieval relevance.
    *   **K-value Tuning:** Experiment with the number of documents (k) retrieved to find an optimal balance between context richness and LLM input limits.
    *   **Alternative Embedding Models:** While `jinaai/jina-embeddings-v2-base-en` is specified, for a real project, exploring other state-of-the-art embedding models could yield better retrieval performance.
*   **RAG Configuration Analysis:**
    *   **Performance Comparison (with/without RAG):** Quantitatively compare the LLM's performance when answering questions with and without the retrieval augmentation component to demonstrate RAG's value.
    *   **Different Retrieval Models:** Compare the RAG system's performance with different retriever configurations (if not limited by assignment constraints).
*   **Post-processing Generated Answers:** Implement rules or another LLM call to refine or reformat generated answers to strictly adhere to the "short answers" requirement and improve exact matching success.
*   **Error Analysis and Debugging:** Conduct a detailed analysis of incorrectly answered questions to identify whether the failure stems from poor retrieval (e.g., irrelevant context) or poor generation (e.g., LLM hallucination or misinterpretation of context). This would guide targeted improvements.

### How to Run and Reproduce

To reproduce the RAG system for this assignment, follow these detailed steps:

1.  **Hugging Face Access Token Setup:**
    *   Navigate to your Hugging Face profile page.
    *   Go to "Settings" -> "Access Tokens".
    *   Click "+ New token", provide an arbitrary name (e.g., "Test"), and select "Fine-grained" as the token type. Ensure read/write permissions are appropriate if needed, but read access is typically sufficient for models.
    *   Copy the generated token.
    *   In your Python environment (e.g., Colab notebook), execute:
        ```python
        from huggingface_hub import login
        hf_token = "YOUR_PASTED_ACCESS_TOKEN_HERE"
        login(token=hf_token, add_to_git_credential=True)
        ```
2.  **Google Colab GPU Configuration:**
    *   Open a new Google Colab notebook.
    *   Go to "Runtime" -> "Change runtime type".
    *   Select "GPU" as the hardware accelerator (e.g., "T4 GPU").
    *   Verify GPU availability by running: `!nvidia-smi`
3.  **Ollama Environment Setup:**
    *   Install `colab-xterm`: `!pip install colab-xterm`
    *   Load the `colabxterm` extension: `%load_ext colabxterm`
    *   Launch an xterm terminal within Colab: `%xterm`
    *   **Inside the launched xterm terminal**, run the following commands:
        *   Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
        *   Pull the required LLM: `ollama pull llama3.2:1b`
        *   Start the Ollama service: `ollama serve` (This command must remain running in the xterm throughout your session).
4.  **Python Code Implementation (LangChain Steps):**
    *   **Initialize LLM and Embedding Model:**
        ```python
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Define model names
        MODEL = "llama3.2:1b"
        EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"

        # Initialize LLM
        llm = Ollama(model=MODEL)

        # Initialize Embedding model
        model_kwargs = {'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        ```
    *   **Load Dataset:**
        ```python
        !wget https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt
        # Read facts into a list 'refs'
        with open("cat-facts.txt", "r", encoding="utf-8") as f:
            refs = [line.strip() for line in f if line.strip()]
        ```
    *   **Prepare Retrieval Database:**
        ```python
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document

        docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]
        vector_store = Chroma.from_documents(docs, embeddings_model)
        retriever = vector_store.as_retriever(search_type="similarity", k=3) # Example k=3, adjust as needed
        ```
    *   **Configure Prompt:**
        ```python
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = """You are an assistant for question-answering tasks about cats.
        Use the following pieces of retrieved context to answer the question concisely.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        {context}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        ```
    *   **Build RAG Chain:**
        ```python
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        ```
    *   **Run and Evaluate:**
        ```python
        # Define the 10 questions and their answers from the assignment (pages 6-7, 22)
        queries = [
            "How much of a day do cats spend sleeping on average?",
            "What is the technical term for a cat's hairball?",
            # ... add all 10 questions ...
        ]
        answers = [
            "2/3",
            "Bezoar",
            # ... add all 10 answers ... (handle lists for multiple correct forms)
        ]

        counts = 0
        for i, query in enumerate(queries):
            response = retrieval_chain.invoke({"input": query}) # Use "input" as defined in ChatPromptTemplate
            print(f"Query: {query}\nResponse: {response['answer']}\n")

            # Evaluation logic (as per page 23)
            if type(answers[i]) == list:
                for ans in answers[i]:
                    if ans.lower() in response['answer'].lower():
                        counts += 1
                        break
            else:
                if answers[i].lower() in response['answer'].lower():
                    counts += 1

        print(f"Correct numbers: {counts}")
        ```

## 📂 Files

- `NLP_HW4_NYCU_111550177.docx`
- `NLP_HW4_NYCU_111550177.py`
- `NTHU-NLP-HW4-RAG.pdf`
- `requirements.txt`

A detailed project description and summary can be found in the [main repository README](../README.md).

