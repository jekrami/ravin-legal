# Multi-Model Persian Contract Analyzer

This project is a sophisticated legal analysis tool designed to interpret and answer questions about Iranian legal contracts. It leverages a multi-model Retrieval-Augmented Generation (RAG) pipeline, local Ollama models, and a modern Gradio interface to provide precise, Persian-first legal analysis.

## Core Features

### 💬 Interactive Document Chat (RAG)
-   **Question-Answer Interface:** Upload legal documents and ask questions in natural Persian language
-   **Multi-Document Support:** Manage multiple documents in a shared knowledge base
-   **Source Attribution:** Every answer includes references to the specific documents and sections used
-   **Persistent Memory:** Chat history and document metadata are saved across sessions
-   **Document Management:** Upload, view, and delete documents with an intuitive interface

### 🔍 Deep Legal Analysis
-   **8-Stage Comprehensive Analysis:** Systematic evaluation through structured analysis passes:
    1. **Structural & Formal Analysis** - Extract contract type, parties, dates, and structure
    2. **Subject Matter & Obligations** - Classify contract subject and party obligations
    3. **Duration & Financial Terms** - Analyze timeframes and monetary conditions
    4. **Termination & Breach** - Evaluate termination clauses and breach remedies
    5. **Risk & Ambiguity Assessment** - Identify risks, ambiguities, and legal gaps
    6. **Iranian Legal Presumptions** - Apply default Iranian legal principles
    7. **Legal Conceptual Model** - Build comprehensive legal framework understanding
    8. **Final Enforceability Evaluation** - Assess overall contract viability
-   **Selective Analysis:** Choose which analysis stages to run based on your needs
-   **Progress Tracking:** Real-time progress visualization with Persian interface
-   **Export Results:** Save analysis results in JSON format and generate Persian reports

### 🤖 Advanced AI Capabilities
-   **Multi-Model Reasoning:** Utilizes three distinct Ollama models in parallel:
    - **Primary Legal Analyst** - Specialized in Persian legal analysis
    - **Secondary Verifier** - Cross-validates findings for accuracy
    - **Synthesizer** - Resolves conflicts and generates coherent, legally conservative answers
-   **RAG Pipeline:** Robust retrieval system with:
    - `pymupdf` for accurate PDF text extraction
    - FAISS vector database for efficient semantic search
    - Hybrid retrieval with keyword boosting for legal terms
    - Multilingual embeddings optimized for Persian text
-   **Persian-First Design:** Optimized for the nuances of Iranian contract law, civil law, and commercial agreements
-   **RTL Support:** Full right-to-left text support with proper Arabic/Persian text reshaping

### 🔒 Privacy & Security
-   **100% Local Processing:** All analysis runs on your local machine
-   **No Cloud Dependencies:** No data is sent to external cloud APIs
-   **Secure Document Handling:** Your sensitive legal documents never leave your system
-   **Offline Capable:** Works completely offline once models are downloaded

## Architecture

The system follows a parallel multi-model architecture:

1.  **User Input:** The user uploads a PDF and asks a question in the Gradio interface.
2.  **RAG Retrieval:** The system retrieves the most relevant text chunks from the document using a combination of semantic search and keyword boosting. Legal terms (مبلغ, تاریخ, ماده, فسخ, etc.) receive boosted relevance scores.
3.  **Parallel Analysis:** The user's question and the retrieved context are sent to two different Ollama models simultaneously:
    *   **Primary Legal Analyst Model** - Specialized in Persian legal analysis (configurable in `config.py`)
    *   **Secondary Verification Model** - Cross-validates the primary analysis (configurable in `config.py`)
4.  **Synthesis:** The responses from the first two models are sent to a third **Synthesizer Model**, which resolves conflicts, preserves factual data, and generates a single, coherent, and legally conservative answer in formal Persian.
5.  **Final Answer:** The synthesized answer, along with source document references, is displayed to the user in the Gradio interface.

**Note:** Model names are configurable in `config.py`. The default setup uses models optimized for Persian legal text analysis.

## Setup and Installation

### 1. Prerequisites

-   **Python 3.9+**
-   **Ollama:** You must have Ollama installed and running. Follow the instructions at [ollama.com](https://ollama.com/).

### 2. Install Ollama Models

You need to pull the required models. The models are configurable in `config.py`, but the default setup requires:

```bash
ollama pull qwen2.5:14b-instruct
ollama pull ravin-gemma3
ollama pull llama3
ollama pull paraphrase-multilingual:278m-mpnet-base-v2-fp16
```

**Note:** You can customize which models to use by editing `config.py`. Make sure the models you specify are compatible with Persian legal text analysis.

### 3. Install Python Dependencies

Clone the repository and install the required Python packages using `pip`:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## How to Run the Application

### 1. Start the Ollama Server

Before running the application, ensure the Ollama server is running in the background.

```bash
ollama serve
```

### 2. Launch the Application

Run the main application from the project's root directory:

```bash
python app_gradio.py
```

The application will start and display a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to access the interface.

### 3. Using the Application

The Gradio interface provides two main tabs:

-   **💬 چت با اسناد (Document Chat):** Upload PDF documents and ask questions about them using the RAG-powered chat interface
-   **🔍 تحلیل عمیق حقوقی (Deep Legal Analysis):** Upload a contract PDF and run comprehensive 8-stage legal analysis with customizable passes

## System Benefits

### 🎯 Accuracy & Reliability
-   **Multi-Model Verification:** Three models work together to cross-validate findings, reducing errors and hallucinations
-   **Structured Analysis:** 8-stage systematic approach ensures comprehensive coverage of all contract aspects
-   **Source Attribution:** Every answer includes document references, enabling fact-checking and verification
-   **Legal Conservatism:** Synthesizer model ensures answers are legally conservative and factually accurate

### ⚡ Performance & Efficiency
-   **Fast Retrieval:** FAISS vector database enables millisecond-level semantic search
-   **Parallel Processing:** Multiple models analyze simultaneously, reducing total analysis time
-   **Selective Analysis:** Run only the analysis stages you need, saving time and resources
-   **Persistent Storage:** Document embeddings are cached, eliminating redundant processing

### 🌐 User Experience
-   **Intuitive Interface:** Clean, modern Gradio interface with full Persian/RTL support
-   **Real-Time Feedback:** Progress indicators and status updates keep you informed
-   **Export Capabilities:** Save analysis results in multiple formats (JSON, Persian text reports)
-   **Document Management:** Easy upload, view, and deletion of documents

### 💼 Professional Use Cases
-   **Contract Review:** Quickly identify risks, ambiguities, and legal issues in contracts
-   **Due Diligence:** Comprehensive analysis of contract enforceability and compliance
-   **Legal Research:** Ask questions about specific clauses, terms, or legal concepts
-   **Document Comparison:** Analyze multiple contracts in a shared knowledge base

## Project Structure

| File                      | Purpose                                                                          |
| ------------------------- | -------------------------------------------------------------------------------- |
| `app_gradio.py`           | **Main entry point** - Gradio web interface with RAG chat and deep analysis     |
| `requirements.txt`        | Python dependencies required for the project                                      |
| `config.py`               | Configuration variables: model names, RAG parameters, API endpoints              |
| `document_processor.py`   | PDF parsing, text extraction, RTL correction, and text chunking                 |
| `rag_pipeline.py`         | FAISS vector store, embeddings, and hybrid retrieval logic with keyword boosting |
| `llm_handler.py`          | Ollama model interactions, parallel execution, and answer synthesis             |
| `legal_analyzer/`         | Deep legal analysis module with 8-stage analysis passes                          |
| `legal_analyzer/passes.py`| Analysis pass definitions and configurations                                     |
| `legal_analyzer/orchestrator.py` | Coordinates multi-stage analysis execution                                      |
| `legal_analyzer/presentation.py` | Generates Persian presentation reports from analysis results                    |
| `vector_db.pkl`           | **Generated:** FAISS index and document chunks (persistent storage)             |
| `rag_chat_history.json`   | **Generated:** Conversation history across sessions                              |
| `documents_metadata.json` | **Generated:** Metadata about uploaded documents                                |
| `deep_analysis_*.json`    | **Generated:** Detailed analysis results in JSON format                          |
| `presentation_fa.txt`     | **Generated:** Persian text report of analysis results                           |
