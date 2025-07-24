"""
Clones a GitHub repository, parses its files using AST-based chunking with tree-sitter,
creates embeddings for these chunks, and stores them in an HNSWlib index and SQLite database.

This script is an updated version to support more granular, semantically relevant code chunks
for an AI-Powered Code Analysis Tool.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable

import gitignore_parser
import hnswlib
import llm
import sqlite_utils
from dotenv import load_dotenv
from tree_sitter import Language, Parser

# Load environment variables from .env file
load_dotenv()


# Define supported non-code file extensions and their types
NON_CODE_FILE_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown", 
    ".txt": "text",
    ".rst": "text",
    ".json": "json",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".cfg": "text",
    ".ini": "text",
    ".conf": "text",
    ".log": "text",
}

# Define AST node types for chunking for different languages
CHUNK_TARGET_NODE_TYPES = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
    "java": ["method_declaration", "class_declaration", "constructor_declaration"],
    "cpp": ["function_definition", "class_specifier", "constructor_definition"],
    "kotlin": [
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "property_declaration",
        "companion_object",
    ],
    # Add more languages and their respective node types
}

# Mapping file extensions to tree-sitter language names and grammar objects
# This needs to be populated with installed grammars
LANGUAGE_GRAMMARS = {}


def initialize_grammars() -> None:
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
    try:
        import tree_sitter_kotlin

        LANGUAGE_GRAMMARS[".kt"] = (Language(tree_sitter_kotlin.language()), "kotlin")
        LANGUAGE_GRAMMARS[".kts"] = (
            Language(tree_sitter_kotlin.language()),
            "kotlin",
        )
    except ImportError:
        print("Warning: tree-sitter-kotlin grammar not found.")
    # Add more languages here


def clone_repo(repo_url: str, target_dir: str) -> None:
    print(f"Cloning repository {repo_url} to {target_dir}...")
    subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    print("Repository cloned successfully.")


def get_gitignore_checker(repo_path: str) -> Callable[[str], bool]:
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if os.path.exists(gitignore_path):
        return gitignore_parser.parse_gitignore(gitignore_path)  # type: ignore[no-any-return]
    return lambda x: False  # If no .gitignore, don't ignore anything


def extract_ast_chunks(
    code_content: str, parser: Parser, language_name: str
) -> list[dict[str, Any]]:
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

        if not chunks and code_content.strip():
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


def extract_markdown_chunks(content: str) -> list[dict[str, Any]]:
    """Extract chunks from markdown files based on headers and sections."""
    chunks = []
    lines = content.split('\n')
    
    # Track current section
    current_section = ""
    current_content = []
    current_start_line = 1
    current_start_byte = 0
    current_byte_pos = 0
    
    for line_num, line in enumerate(lines, 1):
        line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline
        
        # Check if this is a header line (starts with #)
        if line.strip().startswith('#'):
            # Save previous section if it has content
            if current_content and any(l.strip() for l in current_content):
                section_text = '\n'.join(current_content).strip()
                if section_text:
                    chunks.append({
                        "code": section_text,
                        "start_line": current_start_line,
                        "end_line": line_num - 1,
                        "start_byte": current_start_byte,
                        "end_byte": current_byte_pos - 1,
                        "node_type": "markdown_section",
                        "section_title": current_section
                    })
            
            # Start new section
            current_section = line.strip()
            current_content = [line]
            current_start_line = line_num
            current_start_byte = current_byte_pos
        else:
            current_content.append(line)
        
        current_byte_pos += line_bytes
    
    # Add final section
    if current_content and any(l.strip() for l in current_content):
        section_text = '\n'.join(current_content).strip()
        if section_text:
            chunks.append({
                "code": section_text,
                "start_line": current_start_line,
                "end_line": len(lines),
                "start_byte": current_start_byte,
                "end_byte": current_byte_pos,
                "node_type": "markdown_section",
                "section_title": current_section
            })
    
    # If no sections found, treat as single chunk
    if not chunks and content.strip():
        chunks.append({
            "code": content,
            "start_line": 1,
            "end_line": content.count("\n") + 1,
            "start_byte": 0,
            "end_byte": len(bytes(content, "utf8")),
            "node_type": "markdown_file"
        })
    
    return chunks


def extract_text_chunks(content: str) -> list[dict[str, Any]]:
    """Extract chunks from text files based on paragraphs and sections."""
    chunks = []
    
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', content)
    
    current_byte_pos = 0
    current_line = 1
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            current_byte_pos += len(paragraph.encode('utf-8'))
            current_line += paragraph.count('\n')
            continue
            
        paragraph_lines = paragraph.count('\n') + 1
        paragraph_bytes = len(paragraph.encode('utf-8'))
        
        chunks.append({
            "code": paragraph.strip(),
            "start_line": current_line,
            "end_line": current_line + paragraph_lines - 1,
            "start_byte": current_byte_pos,
            "end_byte": current_byte_pos + paragraph_bytes,
            "node_type": "text_paragraph"
        })
        
        current_byte_pos += paragraph_bytes + 2  # +2 for double newline separator
        current_line += paragraph_lines + 1  # +1 for paragraph break
    
    # If no paragraphs found, treat as single chunk
    if not chunks and content.strip():
        chunks.append({
            "code": content,
            "start_line": 1,
            "end_line": content.count("\n") + 1,
            "start_byte": 0,
            "end_byte": len(bytes(content, "utf8")),
            "node_type": "text_file"
        })
    
    return chunks


def extract_json_chunks(content: str) -> list[dict[str, Any]]:
    """Extract chunks from JSON files - treat as single unit."""
    return [{
        "code": content,
        "start_line": 1,
        "end_line": content.count("\n") + 1,
        "start_byte": 0,
        "end_byte": len(bytes(content, "utf8")),
        "node_type": "json_file"
    }]


def extract_non_code_chunks(content: str, file_type: str) -> list[dict[str, Any]]:
    """Extract chunks from non-code files based on file type."""
    if file_type == "markdown":
        return extract_markdown_chunks(content)
    elif file_type == "json":
        return extract_json_chunks(content)
    else:  # text files and others
        return extract_text_chunks(content)


def main() -> None:
    initialize_grammars()
    if not LANGUAGE_GRAMMARS and not NON_CODE_FILE_TYPES:
        print(
            "Error: No file processing capabilities loaded. Please install tree-sitter grammars (e.g., pip install tree-sitter-python). Exiting."
        )
        return
    
    if not LANGUAGE_GRAMMARS:
        print("Warning: No tree-sitter grammars loaded. Only non-code files will be processed.")
    
    print(f"Loaded support for {len(LANGUAGE_GRAMMARS)} code file types and {len(NON_CODE_FILE_TYPES)} non-code file types.")

    db_path = "repo-embeddings.db"
    hnsw_index_path = "repo-embeddings.hnsw"

    db = sqlite_utils.Database(db_path)
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

    model_id = "text-embedding-3-small"  # OpenAI embedding model
    # model_id = "jina-embeddings-v2-base-en" # Alternative if available
    embed_model = llm.get_embedding_model(model_id)
    embedding_dim = None  # We'll get this after the first embedding

    hnsw_index = None
    max_elements = 100000  # Initial capacity, can be resized
    hnsw_ef_construction = 200
    hnsw_M = 16

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
                
                # Check if it's a code file with tree-sitter support
                if file_ext in LANGUAGE_GRAMMARS:
                    lang_obj, lang_name = LANGUAGE_GRAMMARS[file_ext]
                    parser = Parser()
                    parser.language = lang_obj
                    is_code_file = True
                # Check if it's a supported non-code file
                elif file_ext in NON_CODE_FILE_TYPES:
                    file_type = NON_CODE_FILE_TYPES[file_ext]
                    lang_name = file_type
                    is_code_file = False
                else:
                    # Skip unsupported file types
                    # print(f"Skipping file with unsupported extension: {relative_path}")
                    continue

                try:
                    with open(file_path_abs, encoding="utf-8") as f:
                        content = f.read()

                    if not content.strip():  # Skip empty files
                        continue

                    # Extract chunks based on file type
                    if is_code_file:
                        extracted_chunks = extract_ast_chunks(content, parser, lang_name)
                    else:
                        extracted_chunks = extract_non_code_chunks(content, file_type)

                    for chunk_info in extracted_chunks:
                        chunk_id = f"{relative_path}:{chunk_info['start_byte']}:{chunk_info['end_byte']}"
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

        first_chunk_code = all_chunks_to_embed_with_metadata[0]["code"]
        first_embedding = list(embed_model.embed(first_chunk_code))
        embedding_dim = len(first_embedding)
        print(f"Detected embedding dimension: {embedding_dim}")

        hnsw_index = hnswlib.Index(space="cosine", dim=embedding_dim)
        hnsw_index.init_index(
            max_elements=max(len(all_chunks_to_embed_with_metadata), 1),
            ef_construction=hnsw_ef_construction,
            M=hnsw_M,
        )
        hnsw_index.set_num_threads(4)  # Use 4 threads for indexing

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
