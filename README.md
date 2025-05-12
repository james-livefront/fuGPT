# fuGPT: AI-Powered Code Analysis Tool

A tool for automated code analysis using Retrieval-Augmented Generation (RAG) with LLMs to provide feedback on codebases.

## Features

- **Repository Vectorization**: Clone any GitHub repository and convert its code into embeddings
- **Intelligent Code Analysis**: Analyze code against expert-defined evaluation criteria
- **Contextual Query Handling**: Retrieve relevant code snippets based on specific questions
- **Comprehensive Reports**: Generate detailed evaluation reports with strengths and areas for improvement

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/fuGPT.git
   cd fuGPT
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install llm sqlite_utils gitignore_parser
   ```

## Usage

### Vectorize a Repository

```bash
python vectorize-repo.py
```

This script:

- Clones the target repository
- Processes its files according to .gitignore rules
- Creates embeddings stored in an SQLite database

### Analyze a Repository

```bash
python analyze-repo.py
```

This script:

- Reads questions from `questions.txt`
- For each question:
  - Retrieves relevant code snippets from the database
  - Sends them to an LLM along with the question
  - Saves the generated answer
- Compiles all answers into a final report saved as `final_code_evaluation.md`

## Project Structure

```
fuGPT/
├── README.md                  # This file
├── vectorize-repo.py          # Repository vectorization script
├── analyze-repo.py            # Code analysis script
├── analyze-repo-flow.mmd      # Analysis workflow diagram
├── system-prompt.txt          # LLM system prompt for code evaluation
├── questions.txt              # Code evaluation criteria
├── final_code_evaluation.md   # Generated evaluation report (after running analysis)
└── repo-embeddings.db         # SQLite database storing code embeddings
```

## How It Works

1. **Repository Vectorization**: The tool clones a GitHub repository, parses its files, and creates embeddings using LLM, storing them in a SQLite database.

2. **Question-Based Analysis**: For each question in `questions.txt`, the tool:
   - Queries the database for relevant code snippets
   - Combines snippets with the system prompt and question
   - Executes an LLM query to generate an expert analysis

3. **Report Generation**: All answers are compiled into a comprehensive report with strengths and areas for improvement.

## Customization

- Modify `questions.txt` to add or update evaluation criteria
- Edit `system-prompt.txt` to change the LLM's role and evaluation approach
- Adjust parameters in the scripts for different embedding models or number of results

## License

[Insert license information here]

## Contributing

Contributions, issues, and feature requests are welcome!
