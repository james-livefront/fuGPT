"""
Analyzes a codebase by retrieving relevant code chunks using a hybrid RAG pipeline
(AST-based chunks, HNSWlib, BM25, Reranker, optional HyDE) and querying an LLM.

This script is an updated version for the AI-Powered Code Analysis Tool.
"""

import os
import re
import subprocess
import tempfile
import time

# Set tokenizers parallelism to avoid warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# HNSWlib import (assuming it will be installed in the target environment)
import hnswlib
import llm
import sqlite_utils

# BM25 import (assuming it will be installed)
from rank_bm25 import BM25Okapi

# Sentence Transformers import for Reranker (assuming it will be installed)
from sentence_transformers import CrossEncoder

# --- Configuration ---
db_path = "repo-embeddings.db"  # Path to the SQLite DB created by vectorize-repo.py
hnsw_index_path = "repo-embeddings.hnsw"  # Path to the HNSWlib index

# Embedding model for queries (should match the one used for vectorizing chunks)
# This will be loaded via the llm library
QUERY_EMBED_MODEL_ID = "text-embedding-3-small"  # OpenAI embedding model, ensure consistency with vectorize-repo.py
# QUERY_EMBED_MODEL_ID = "text-embedding-3-small"

# Reranker model
RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A common lightweight reranker
)

# HyDE configuration
USE_HYDE = True  # Set to False to disable HyDE
HYDE_LLM_MODEL_ID = "gpt-4o-mini"  # A fast OpenAI LLM for generating hypothetical documents
# Or use a specific llm library model if not using shell command for HyDE

# Retrieval parameters
SEMANTIC_SEARCH_TOP_K = 15  # Number of results from HNSWlib
BM25_TOP_K = 15  # Number of results from BM25
RERANKER_TOP_K = 5  # Final number of chunks to feed to LLM
MAX_CHARS_PER_CHUNK_IN_CONTEXT = (
    2000  # Max characters from each chunk to add to LLM context
)

# --- Global Variables --- (Load models once)
query_embed_model = None
hnsw_index = None
chunks_db = None
reranker_model = None
bm25_corpus = []
bm25_index = None
all_db_chunks = []  # To store all chunks from DB for BM25 and metadata lookup


def initialize_models_and_data():
    """Load models, HNSW index, and SQLite data once."""
    global \
        query_embed_model, \
        hnsw_index, \
        chunks_db, \
        reranker_model, \
        bm25_corpus, \
        bm25_index, \
        all_db_chunks

    print("Initializing models and data...")

    # Load query embedding model via llm library
    try:
        query_embed_model = llm.get_embedding_model(QUERY_EMBED_MODEL_ID)
        # Test embedding to get dimension for HNSWlib if not known, or load index and get it
    except Exception as e:
        print(f"Error loading query embedding model '{QUERY_EMBED_MODEL_ID}': {e}")
        print("Please ensure the llm tool is configured and the model is available.")
        raise

    # Connect to SQLite database
    if not os.path.exists(db_path):
        print(
            f"Error: SQLite database not found at {db_path}. Please run vectorize-repo.py first."
        )
        raise FileNotFoundError(db_path)
    chunks_db = sqlite_utils.Database(db_path)
    if "chunks" not in chunks_db.table_names():
        print(f"Error: 'chunks' table not found in {db_path}.")
        raise Exception("'chunks' table missing")

    # Load all chunks from DB for BM25 and metadata lookup
    all_db_chunks = list(chunks_db["chunks"].rows)
    if not all_db_chunks:
        print("Warning: No chunks found in the database.")
        # return # Or raise error, depending on desired behavior

    # Load HNSWlib index
    if not os.path.exists(hnsw_index_path):
        print(
            f"Error: HNSWlib index not found at {hnsw_index_path}. Please run vectorize-repo.py first."
        )
        raise FileNotFoundError(hnsw_index_path)

    # To load the HNSW index, we need the dimension.
    # We assume the first embedding from the model gives us the correct dimension.
    # This is a bit of a workaround; ideally, dimension is stored or known.
    try:
        dummy_embedding = list(query_embed_model.embed("test"))
        embedding_dim = len(dummy_embedding)
        print(f"Deduced embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"Could not deduce embedding dimension: {e}")
        # Fallback or error, as HNSWlib needs this. For now, let's assume a common one if it fails.
        # This part should be robust, e.g. by storing dim during vectorization.
        print(
            "Attempting to load HNSW index without pre-deducing dimension. This might fail if dim is incorrect."
        )
        embedding_dim = (
            768  # A common default, but might be wrong for Jina or OpenAI new models
        )

    hnsw_index = hnswlib.Index(space="cosine", dim=embedding_dim)
    hnsw_index.load_index(hnsw_index_path)
    print(f"HNSWlib index loaded with {hnsw_index.get_current_count()} elements.")

    # Initialize BM25 Index
    if all_db_chunks:
        bm25_corpus_texts = [chunk["code"] for chunk in all_db_chunks]
        tokenized_corpus = [doc.split(" ") for doc in bm25_corpus_texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        print("BM25 index initialized.")
    else:
        print("BM25 index not initialized as no chunks were loaded.")

    # Load Reranker model
    try:
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"Reranker model '{RERANKER_MODEL_NAME}' loaded.")
    except Exception as e:
        print(f"Error loading reranker model '{RERANKER_MODEL_NAME}': {e}")
        print("Reranking will be skipped if model is not available.")
        reranker_model = None  # Allow proceeding without reranker if it fails

    print("Initialization complete.")


# Function to generate hypothetical document (HyDE)
def generate_hypothetical_document(query):
    if not USE_HYDE:
        return query  # Return original query if HyDE is disabled

    print(f"Generating hypothetical document for query: {query[:100]}...")
    try:
        # Using llm CLI for simplicity, as in the original analyze-repo.py
        # This assumes the llm CLI is installed and configured.
        hyde_prompt = f"Write a short, ideal code snippet or explanation that would perfectly answer the following question about a codebase: {query}"
        # Create a temporary file for the prompt
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as tmp_prompt_file:
            tmp_prompt_file.write(hyde_prompt)
            tmp_prompt_file_name = tmp_prompt_file.name

        with tempfile.NamedTemporaryFile(
            mode="r", delete=False, suffix=".txt"
        ) as tmp_output_file:
            tmp_output_file_name = tmp_output_file.name

        # Use a simpler/faster model for HyDE if possible
        # The model should be available to the `llm` tool
        shell_cmd = f"cat {tmp_prompt_file_name} | llm -m {HYDE_LLM_MODEL_ID} > {tmp_output_file_name}"
        subprocess.run(shell_cmd, shell=True, check=True, timeout=60)

        with open(tmp_output_file_name, encoding="utf-8") as f:
            hypothetical_doc = f.read().strip()

        print(f"Generated hypothetical document (HyDE): {hypothetical_doc[:200]}...")
        return hypothetical_doc
    except Exception as e:
        print(f"Error generating hypothetical document: {e}. Using original query.")
        return query
    finally:
        if os.path.exists(tmp_prompt_file_name):
            os.remove(tmp_prompt_file_name)
        if os.path.exists(tmp_output_file_name):
            os.remove(tmp_output_file_name)


# Function to get relevant code chunks
def get_relevant_chunks(query):
    if not hnsw_index or not query_embed_model or not all_db_chunks:
        print("Error: Models or data not initialized properly.")
        return []

    # 1. (Optional) HyDE: Generate hypothetical document and embed it
    search_text = generate_hypothetical_document(query)
    query_embedding = list(query_embed_model.embed(search_text))

    # 2. Semantic Search (HNSWlib)
    print(f"Performing semantic search for: {search_text[:100]}...")
    labels, distances = hnsw_index.knn_query(query_embedding, k=SEMANTIC_SEARCH_TOP_K)
    semantic_results_ids = labels[0]  # HNSWlib returns embedding_id
    # print(f"Semantic search raw results (embedding_ids): {semantic_results_ids}")

    # 3. Keyword Search (BM25)
    print(f"Performing BM25 search for: {query[:100]}...")
    tokenized_query = query.lower().split(" ")  # Use original query for BM25
    bm25_scores = bm25_index.get_scores(tokenized_query) if bm25_index else []

    # Get top BM25 results (indices in all_db_chunks)
    bm25_top_indices = (
        sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[
            :BM25_TOP_K
        ]
        if len(bm25_scores) > 0
        else []
    )
    bm25_results_embedding_ids = [
        all_db_chunks[i]["embedding_id"] for i in bm25_top_indices
    ]
    # print(f"BM25 search raw results (embedding_ids): {bm25_results_embedding_ids}")

    # 4. Combine and Deduplicate Results (using embedding_id as unique key for chunks)
    combined_embedding_ids = list(
        dict.fromkeys(list(semantic_results_ids) + bm25_results_embedding_ids)
    )
    # print(f"Combined (deduplicated) embedding_ids: {combined_embedding_ids}")

    # Retrieve chunk details for reranking
    candidate_chunks_for_reranking = []
    for emb_id in combined_embedding_ids:
        # Find the chunk in all_db_chunks by embedding_id
        # This is inefficient if all_db_chunks is very large; a dict lookup would be better.
        # For now, linear scan. In vectorize, ensure embedding_id is a direct index or make a map.
        chunk_detail = next(
            (chunk for chunk in all_db_chunks if chunk["embedding_id"] == emb_id), None
        )
        if chunk_detail:
            candidate_chunks_for_reranking.append(chunk_detail)

    if not candidate_chunks_for_reranking:
        print("No candidates found after semantic and BM25 search.")
        return []

    # 5. Rerank (Cross-Encoder)
    if reranker_model:
        print(f"Reranking {len(candidate_chunks_for_reranking)} candidates...")
        # Reranker expects pairs of (query, passage)
        rerank_pairs = [
            [query, chunk["code"]] for chunk in candidate_chunks_for_reranking
        ]
        rerank_scores = reranker_model.predict(rerank_pairs)
        
        # Apply heuristic boosts for documentation queries
        documentation_keywords = ["readme", "documentation", "docs", "setup", "install", "getting started"]
        if any(keyword in query.lower() for keyword in documentation_keywords):
            for i, chunk in enumerate(candidate_chunks_for_reranking):
                # Boost README and other documentation files
                if chunk["file_path"].lower() in ["readme.md", "readme.txt", "docs.md"] or \
                   chunk["language"] == "markdown" or \
                   "readme" in chunk["file_path"].lower():
                    rerank_scores[i] += 2.0  # Significant boost for documentation files
                    print(f"Boosted documentation file: {chunk['file_path']}")
                # Boost other common documentation patterns
                elif chunk["file_path"].lower().endswith((".md", ".txt", ".rst")) and \
                     any(doc_word in chunk["file_path"].lower() for doc_word in ["doc", "guide", "help", "install"]):
                    rerank_scores[i] += 1.0  # Moderate boost for other doc files

        # Sort chunks by reranker scores
        reranked_chunks_with_scores = list(
            zip(candidate_chunks_for_reranking, rerank_scores)
        )
        reranked_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

        final_chunks_data = [
            chunk_score[0]
            for chunk_score in reranked_chunks_with_scores[:RERANKER_TOP_K]
        ]
        # print(f"Reranked top {RERANKER_TOP_K} chunk IDs (original file:line): {[f'{c["file_path"]}:{c["start_line"]}' for c in final_chunks_data]}")
    else:
        print(
            "Skipping reranking as model is not available. Using combined results directly."
        )
        final_chunks_data = candidate_chunks_for_reranking[:RERANKER_TOP_K]

    return final_chunks_data


# Function to query the LLM with a question and context
def ask_about_codebase(question, aspect_for_retrieval):
    retrieved_chunks = get_relevant_chunks(aspect_for_retrieval)

    if not retrieved_chunks:
        context_str = "No relevant code chunks found."
    else:
        context_items = []
        for chunk_data in retrieved_chunks:
            code_snippet = chunk_data["code"]
            if len(code_snippet) > MAX_CHARS_PER_CHUNK_IN_CONTEXT:
                code_snippet = (
                    code_snippet[:MAX_CHARS_PER_CHUNK_IN_CONTEXT]
                    + "\n...[truncated]..."
                )
            context_items.append(
                f"FILE: {chunk_data['file_path']} (lines {chunk_data['start_line']}-{chunk_data['end_line']}, type: {chunk_data['node_type']})\nLANGUAGE: {chunk_data['language']}\nCODE:\n{code_snippet}\n---"
            )
        context_str = "\n".join(context_items)

    # Read the system prompt
    # Load system prompt from config directory
    try:
        with open("config/system-prompt.txt", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        print("Warning: config/system-prompt.txt not found. Using a default prompt.")
        system_prompt = "You are a helpful AI assistant. Analyze the provided code context to answer the question."

    prompt = f"{system_prompt}\n\nRELEVANT CODE CONTEXT:\n{context_str}\n\nQUESTION: {question}\n\nPlease provide strengths and areas for improvement based on the code context and question."

    # Save to temp file (as in original script)
    temp_file_path = "temp_query.txt"
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    sanitized_question = (
        re.sub(r"[^\w\s]", "", question[:40]).strip().replace(" ", "_").lower()
    )
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    output_file_path = f"outputs/answer_{sanitized_question}.md"

    # Execute the LLM query using shell command (as in original script)
    # Ensure the LLM model used here is powerful enough for analysis
    llm_model_for_analysis = (
        "gpt-4o"  # Using OpenAI's GPT-4 for analysis
    )
    print(f"Asking LLM ({llm_model_for_analysis}): {question[:60]}...")
    shell_cmd = (
        f"cat {temp_file_path} | llm -m {llm_model_for_analysis} > {output_file_path}"
    )

    try:
        os.system(shell_cmd)
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            print(f"✓ Answer generated: {output_file_path}")
            with open(output_file_path, encoding="utf-8") as f:
                return f.read()
        else:
            print("✗ Failed to generate answer or output file is empty.")
            return f"[Error: Could not generate answer for: {question}]"
    except Exception as e:
        print(f"✗ Error during LLM execution: {e}")
        return f"[Error executing LLM for: {question} - {e}]"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def main():
    try:
        initialize_models_and_data()
    except Exception as e:
        print(f"Failed to initialize. Exiting. Error: {e}")
        return

    # Read questions from config directory
    try:
        with open("config/questions.txt", encoding="utf-8") as f:
            questions = [q.strip() for q in f.read().splitlines() if q.strip()]
    except FileNotFoundError:
        print("Error: config/questions.txt not found. Please create it with questions to ask.")
        return

    if not questions:
        print("No questions found in config/questions.txt. Exiting.")
        return

    all_answers = []
    report_title = input(
        "Enter a title for the final evaluation report (e.g., MyProject Code Evaluation): "
    )
    if not report_title:
        report_title = "Codebase Evaluation Report v2"

    for i, question_text in enumerate(questions):
        print(
            f"\nProcessing question {i + 1}/{len(questions)}: {question_text[:80]}..."
        )

        # Use the full question as the aspect for retrieval, or a summary if preferred
        # The HyDE step will attempt to rephrase it anyway if enabled.
        aspect = question_text

        answer = ask_about_codebase(question_text, aspect)
        all_answers.append(f"## Question {i + 1}: {question_text}\n\n{answer}\n")

        if i < len(questions) - 1:
            print("Waiting a few seconds before next question...")
            time.sleep(3)  # Delay as in original

    final_report_content = f"# {report_title}\n\n" + "\n".join(all_answers)
    final_report_filename = "final_code_evaluation.md"
    with open(final_report_filename, "w", encoding="utf-8") as f:
        f.write(final_report_content)

    print(f"\nEvaluation complete! Final report saved to {final_report_filename}")


if __name__ == "__main__":
    main()
