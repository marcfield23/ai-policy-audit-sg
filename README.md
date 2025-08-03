# Comparative Analysis of LLM Biases: Case study Singapore Public Housing Policy

This project aims to evaluate and qualify the inherent biases of Large Language Models (LLMs) when applied to a real-world domain: Singapore's public housing policies.

### Key Goal

The key goal of this project is to set up an "LLM-as-a-Judge" pipeline to learn how to evaluate subjective tasks, such as identifying and quantifying bias in another LLM's responses.

### How It Works

The system uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on a controlled set of official HDB policy documents. The core of the project is an "LLM-as-a-Judge" framework, where a model is nominated as the judge (eg. Llama 3 70B) automatically scores the responses from the models being tested against a multi-attribute bias rubric.

* **Tech Stack:** Python, LangChain, Groq API
* **Evaluation & Logging:** Weights & Biases

### How to Replicate This Experiment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marcfield23/llm-ops-bias-eval.git
    ```

2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Add your API keys:**
    * Create a `.env` file in the root directory.
    * Add your Groq API key: `GROQ_API_KEY="..."`.

4.  **Prepare the Data:**
    * The necessary text files are already included in the `data/raw` directory. To rebuild the knowledge base with your own data, simply add or remove `.txt` files in this folder.
    * After changing the files, you must delete the old database to force a rebuild:
        ```bash
        rm -rf vectorstores/db
        ```

5.  **Run the Full Evaluation:**
    * Execute the main evaluation script. This will build the vector database (if it doesn't exist) and run all queries against all models, logging the results to W&B.
        ```bash
        python src/run_evaluation.py
        ```

6.  **Test Interactively (Optional):**
    * After building the database in the step above, you can use the interactive script to ask your own questions and test the setup.
        ```bash
        python src/interactive_query.py
        ```

### Customizing the Experiment

You can easily adapt this project to test different scenarios.

* **To change the knowledge base:** Add, remove, or edit the `.txt` files inside the `data/raw` directory. Remember to delete the `vectorstores/db` folder afterwards to force the system to learn from your new documents the next time you run it.

* **To change the test questions:** Edit the `data/query_bank.json` file. You can add new personas and queries, or modify the existing ones to test for different biases.
