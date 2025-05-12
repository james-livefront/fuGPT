"""
Clones a GitHub repository to a target directory and checks if files should be included based on the repository's .gitignore file.

Creates a new SQLite database and a collection within it, using the specified embedding model, to store the contents and metadata of the cloned repository files.

Walks through the cloned repository, skipping the .git directory and any files that match the .gitignore patterns, and embeds the contents of the remaining files in batches into the collection.

The resulting vector database is stored at the specified path.
"""
import llm
import sqlite_utils
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import gitignore_parser

# Function to clone a GitHub repository
def clone_repo(repo_url, target_dir):
    print(f"Cloning repository {repo_url} to {target_dir}...")
    subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    print("Repository cloned successfully.")

# Function to check if path should be included (not in .gitignore)
def get_gitignore_checker(repo_path):
    gitignore_path = os.path.join(repo_path, '.gitignore')
    if os.path.exists(gitignore_path):
        return gitignore_parser.parse_gitignore(gitignore_path)
    return lambda x: False  # If no .gitignore, don't ignore anything

# Create or connect to the database
db_path = "repo-embeddings.db"
db = sqlite_utils.Database(db_path)

# Set up the collection
collection_name = "chicago_artwork"
model_id = "3-small"  # You can change this to your preferred embedding model

# Check if collection exists and remove if needed
if llm.Collection.exists(db, collection_name):
    print(f"Collection '{collection_name}' already exists. Deleting...")
    collection = llm.Collection(collection_name, db)
    collection.delete()
    print(f"Collection '{collection_name}' deleted.")

# Create a new collection
collection = llm.Collection(collection_name, db, model_id=model_id)
print(f"Created new collection '{collection_name}' using model '{model_id}'")

# Set up a temporary directory for the cloned repository
temp_dir = tempfile.mkdtemp()
try:
    # Clone the repository
    repo_url = "https://github.com/willdenne/chicago-artwork"
    clone_repo(repo_url, temp_dir)

    # Get gitignore checker
    is_ignored = get_gitignore_checker(temp_dir)

    # Walk through the repository
    files_to_embed = []
    print("Scanning repository files...")

    for root, dirs, files in os.walk(temp_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, temp_dir)

            # Skip files that match gitignore patterns
            if is_ignored(file_path):
                continue

            try:
                # Try to read all files as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Get file extension
                _, ext = os.path.splitext(file)

                # Get file metadata
                file_stat = os.stat(file_path)
                metadata = {
                    "file_extension": ext,
                    "file_size": file_stat.st_size,
                    "last_modified": file_stat.st_mtime
                }

                # Use relative path as ID
                files_to_embed.append((relative_path, content, metadata))

            except UnicodeDecodeError:
                print(f"Skipping binary file: {relative_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Found {len(files_to_embed)} files to embed")

    # Embed files in batches
    batch_size = 20  # Adjust based on your embedding model and file sizes
    for i in range(0, len(files_to_embed), batch_size):
        batch = files_to_embed[i:i+batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(files_to_embed) + batch_size - 1)//batch_size}")
        collection.embed_multi_with_metadata(batch, store=True)

    print(f"Embedded {collection.count()} files from the repository")
    print(f"Vector database created at: {os.path.abspath(db_path)}")

finally:
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print("Temporary repository files removed")