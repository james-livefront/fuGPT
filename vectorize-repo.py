"""
Clones a GitHub repository, parses its files using AST-based chunking with tree-sitter,
creates embeddings for these chunks, and stores them in an HNSWlib index and SQLite database.

This script is an updated version to support more granular, semantically relevant code chunks
for an AI-Powered Code Analysis Tool.
"""

import os
import shutil
import subprocess
import tempfile

import gitignore_parser

# HNSWlib import (assuming it will be installed in the target environment)
import hnswlib
import llm
import sqlite_utils

# Tree-sitter imports
from tree_sitter import Language, Parser

# Assuming tree-sitter grammars are installed as separate packages
# e.g., tree_sitter_python, tree_sitter_javascript, etc.
# We will dynamically load them based on file extension.


# Configuration for tree-sitter and chunking
# Define AST node types for chunking for different languages
# This is a simplified example; a more robust solution would have a more comprehensive list
# and potentially a way to configure this externally.
CHUNK_TARGET_NODE_TYPES = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
    "java": ["method_declaration", "class_declaration", "constructor_declaration"],
    "cpp": ["function_definition", "class_specifier", "constructor_definition"],
    # Add more languages and their respective node types
}

# Mapping file extensions to tree-sitter language names and grammar objects
# This needs to be populated with installed grammars
LANGUAGE_GRAMMARS = {}


def initialize_grammars():
    """Dynamically loads available tree-sitter grammars."""
    global LANGUAGE_GRAMMARS
    try:
        import tree_sitter_python

        LANGUAGE_GRAMMARS[".py"] = (Language(tree_sitter_python.language()), "python")
    except ImportError:
        print("Warning: tree-sitter-python grammar not found.")
    try:
        import tree_sitter_javascript

        LANGUAGE_GRAMMARS[".js"] = (
            Language(tree_sitter_javascript.language()),
            "javascript",
        )
    except ImportError:
        print("Warning: tree-sitter-javascript grammar not found.")
    try:
        import tree_sitter_java

        LANGUAGE_GRAMMARS[".java"] = (Language(tree_sitter_java.language()), "java")
    except ImportError:
        print("Warning: tree-sitter-java grammar not found.")
    try:
        import tree_sitter_cpp

        LANGUAGE_GRAMMARS[".cpp"] = (Language(tree_sitter_cpp.language()), "cpp")
        LANGUAGE_GRAMMARS[".h"] = (
            Language(tree_sitter_cpp.language()),
            "cpp",
        )  # Also for headers
        LANGUAGE_GRAMMARS[".hpp"] = (Language(tree_sitter_cpp.language()), "cpp")
        LANGUAGE_GRAMMARS[".cc"] = (Language(tree_sitter_cpp.language()), "cpp")
    except ImportError:
        print("Warning: tree-sitter-cpp grammar not found.")
    # Add more languages here


# Function to clone a GitHub repository
def clone_repo(repo_url, target_dir):
    print(f"Cloning repository {repo_url} to {target_dir}...")
    subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    print("Repository cloned successfully.")


# Function to check if path should be included (not in .gitignore)
def get_gitignore_checker(repo_path):
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if os.path.exists(gitignore_path):
        return gitignore_parser.parse_gitignore(gitignore_path)
    return lambda x: False  # If no .gitignore, don't ignore anything


# Function to extract AST-based chunks from code
def extract_ast_chunks(code_content, parser, language_name):
    chunks = []
    try:
        tree = parser.parse(bytes(code_content, "utf8"))
        root_node = tree.root_node

        target_nodes = CHUNK_TARGET_NODE_TYPES.get(language_name, [])
        if not target_nodes:
            # Fallback for unsupported languages or if no specific nodes defined: chunk whole file
            chunks.append(
                {
                    "code": code_content,
                    "start_line": 1,
                    "end_line": code_content.count("\n") + 1,
                    "start_byte": 0,
                    "end_byte": len(bytes(code_content, "utf8")),
                    "node_type": "file",
                }
            )
            return chunks

        queue = [root_node]
        visited_nodes = set()

        while queue:
            node = queue.pop(0)
            if node.id in visited_nodes:
                continue
            visited_nodes.add(node.id)

            if node.type in target_nodes:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_code = code_content[node.start_byte : node.end_byte]
                chunks.append(
                    {
                        "code": chunk_code,
                        "start_line": start_line,
                        "end_line": end_line,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "node_type": node.type,
                    }
                )
                # Avoid re-processing children of an already extracted chunk of primary interest
                # This simple approach might need refinement for overlapping or nested structures if desired
                continue

            for child in node.children:
                queue.append(child)

        if (
            not chunks and code_content.strip()
        ):  # If no target nodes found, but file has content
            chunks.append(
                {
                    "code": code_content,
                    "start_line": 1,
                    "end_line": code_content.count("\n") + 1,
                    "start_byte": 0,
                    "end_byte": len(bytes(code_content, "utf8")),
                    "node_type": "file_fallback",
                }
            )

    except Exception as e:
        print(f"Error parsing or chunking code for language {language_name}: {e}")
        # Fallback: add the whole file content as a single chunk
        chunks.append(
            {
                "code": code_content,
                "start_line": 1,
                "end_line": code_content.count("\n") + 1,
                "start_byte": 0,
                "end_byte": len(bytes(code_content, "utf8")),
                "node_type": "file_error_fallback",
            }
        )
    return chunks


def main():
    initialize_grammars()
    if not LANGUAGE_GRAMMARS:
        print(
            "Error: No tree-sitter grammars loaded. Please install them (e.g., pip install tree-sitter-python). Exiting."
        )
        return

    # Database and HNSWlib setup
    db_path = "repo-embeddings.db"
    hnsw_index_path = "repo-embeddings.hnsw"

    db = sqlite_utils.Database(db_path)
    # The 'chunks' table will store metadata and the actual code chunk
    # Embeddings will be in HNSWlib, IDs will link them.
    if "chunks" in db.table_names():
        db["chunks"].drop()
    chunks_table = db["chunks"].create(
        {
            "id": str,  # Unique ID for the chunk (e.g., filepath:start_byte:end_byte)
            "file_path": str,
            "language": str,
            "node_type": str,
            "start_line": int,
            "end_line": int,
            "start_byte": int,
            "end_byte": int,
            "code": str,  # The actual code chunk
            "embedding_id": int,  # ID in HNSWlib index
        },
        pk="id",
    )

    # Embedding model setup (using llm library as in original)
    # Ensure the model_id is appropriate for code embeddings
    model_id = "text-embedding-3-small"  # OpenAI embedding model
    # model_id = "jina-embeddings-v2-base-en" # Alternative if available
    embed_model = llm.get_embedding_model(model_id)
    embedding_dim = None  # We'll get this after the first embedding

    # HNSWlib index setup (will be initialized after knowing embedding_dim)
    hnsw_index = None
    max_elements = 100000  # Initial capacity, can be resized
    hnsw_ef_construction = 200
    hnsw_M = 16

    # Temporary directory for cloning
    temp_dir = tempfile.mkdtemp()
    try:
        repo_url = input("Enter GitHub repository URL to vectorize: ")
        if not repo_url:
            repo_url = (
                "https://github.com/willdenne/chicago-artwork"  # Default for testing
            )
            print(f"No URL entered, using default: {repo_url}")

        clone_repo(repo_url, temp_dir)
        is_ignored = get_gitignore_checker(temp_dir)

        all_chunks_to_embed_with_metadata = []
        current_embedding_id = 0

        print("Scanning repository files and extracting chunks...")
        for root, dirs, files in os.walk(temp_dir):
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                file_path_abs = os.path.join(root, file)
                relative_path = os.path.relpath(file_path_abs, temp_dir)

                if is_ignored(file_path_abs):
                    # print(f"Skipping ignored file: {relative_path}")
                    continue

                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in LANGUAGE_GRAMMARS:
                    # print(f"Skipping file with unsupported extension: {relative_path}")
                    continue

                lang_obj, lang_name = LANGUAGE_GRAMMARS[file_ext]
                parser = Parser()
                parser.set_language(lang_obj)

                try:
                    with open(file_path_abs, encoding="utf-8") as f:
                        content = f.read()

                    if not content.strip():  # Skip empty files
                        continue

                    extracted_code_chunks = extract_ast_chunks(
                        content, parser, lang_name
                    )

                    for chunk_info in extracted_code_chunks:
                        chunk_id = f"{relative_path}:{chunk_info['start_byte']}:{chunk_info['end_byte']}"
                        # Prepare data for SQLite and for embedding
                        chunk_data_for_db = {
                            "id": chunk_id,
                            "file_path": relative_path,
                            "language": lang_name,
                            "node_type": chunk_info["node_type"],
                            "start_line": chunk_info["start_line"],
                            "end_line": chunk_info["end_line"],
                            "start_byte": chunk_info["start_byte"],
                            "end_byte": chunk_info["end_byte"],
                            "code": chunk_info["code"],
                            "embedding_id": current_embedding_id,
                        }
                        all_chunks_to_embed_with_metadata.append(chunk_data_for_db)
                        current_embedding_id += 1

                except UnicodeDecodeError:
                    print(f"Skipping binary or non-UTF-8 file: {relative_path}")
                except Exception as e:
                    print(f"Error processing file {relative_path}: {e}")

        print(f"Found {len(all_chunks_to_embed_with_metadata)} chunks to embed.")

        if not all_chunks_to_embed_with_metadata:
            print("No code chunks found to embed. Exiting.")
            return

        # Get embedding dimension from the first chunk
        first_chunk_code = all_chunks_to_embed_with_metadata[0]["code"]
        first_embedding = list(embed_model.embed(first_chunk_code))
        embedding_dim = len(first_embedding)
        print(f"Detected embedding dimension: {embedding_dim}")

        # Initialize HNSWlib index now that we have the dimension
        hnsw_index = hnswlib.Index(space="cosine", dim=embedding_dim)
        hnsw_index.init_index(
            max_elements=max(len(all_chunks_to_embed_with_metadata), 1),
            ef_construction=hnsw_ef_construction,
            M=hnsw_M,
        )
        hnsw_index.set_num_threads(4)  # Use 4 threads for indexing

        # Embed and store chunks
        print(f"Embedding {len(all_chunks_to_embed_with_metadata)} chunks...")
        embeddings_for_hnsw = []
        ids_for_hnsw = []

        for i, chunk_data in enumerate(all_chunks_to_embed_with_metadata):
            if i == 0:  # We already embedded the first one
                embedding = first_embedding
            else:
                embedding = list(embed_model.embed(chunk_data["code"]))

            embeddings_for_hnsw.append(embedding)
            ids_for_hnsw.append(chunk_data["embedding_id"])

            # Store metadata in SQLite
            chunks_table.insert(chunk_data, pk="id", replace=True)

            if (i + 1) % 100 == 0:
                print(
                    f"Processed {i + 1}/{len(all_chunks_to_embed_with_metadata)} chunks..."
                )

        print("Adding embeddings to HNSWlib index...")
        hnsw_index.add_items(embeddings_for_hnsw, ids_for_hnsw)
        hnsw_index.save_index(hnsw_index_path)

        print(f"Successfully embedded {chunks_table.count} chunks.")
        print(f"SQLite database saved at: {os.path.abspath(db_path)}")
        print(f"HNSWlib index saved at: {os.path.abspath(hnsw_index_path)}")

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
