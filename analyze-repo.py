"""
Generates a report evaluating the code in a codebase based on questions asked about the codebase.

The report includes:
- Retrieving relevant code snippets from a database based on the question asked
- Combining the retrieved code snippets with a system prompt and the question
- Executing an LLM model to generate an answer to the question
- Saving the generated answer to a file
- Combining all answers into a final report and saving it to a file

The report is intended to provide insights and feedback on the codebase to help improve it.
"""
import llm
import sqlite_utils
import os
import time
import re

# Connect to your existing database
db_path = "repo-embeddings.db"
db = sqlite_utils.Database(db_path)
collection_name = "chicago_artwork"
collection = llm.Collection(collection_name, db)

# Read the system prompt
with open("system-prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# Read the questions
with open("questions.txt", "r", encoding="utf-8") as f:
    questions = [q.strip() for q in f.read().splitlines() if q.strip()]

# Function to get relevant code for a specific topic
def get_relevant_code(query, num_results=3, max_chars_per_file=1500):
    results = collection.similar(query, number=num_results)
    context = []
    for entry in results:
        if entry.content:
            # Truncate very large files
            content = entry.content
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + "\n...[truncated]..."
            context.append(f"FILE: {entry.id}\n{content}\n")

    return "\n".join(context)

# Function to query the LLM with a question and context
def ask_about_codebase(question, aspect):
    # Get relevant code based on the aspect/question
    context = get_relevant_code(aspect)

    # Combine system prompt, context, and question
    prompt = f"{system_prompt}\n\nRELEVANT CODE:\n{context}\n\nQUESTION: {question}\n\nPlease provide strengths and areas for improvement."

    # Save to temp file
    temp_file = f"temp_query.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # Output file for this question
    sanitized_question = re.sub(r'[^\w\s]', '', question[:40]).strip().replace(' ', '_').lower()
    output_file = f"answer_{sanitized_question}.txt"

    # Execute the LLM query using shell command
    print(f"Asking: {question[:60]}...")
    shell_cmd = f"cat {temp_file} | llm -m deepseek-coder-v2:latest > {output_file}"
    os.system(shell_cmd)

    # Check if output was generated
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"✓ Answer generated")
        with open(output_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"✗ Failed to generate answer")
        return f"[Error: Could not generate answer for: {question}]"

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Process each question
all_answers = []
for i, question in enumerate(questions):
    print(f"\nProcessing question {i+1}/{len(questions)}")

    # Extract key aspect from the question to use for retrieval
    # This is a simple approach - you might want to enhance this
    aspect = ' '.join(question.split()[:6])

    # Get the answer
    answer = ask_about_codebase(question, aspect)
    all_answers.append(f"## Question {i+1}: {question}\n\n{answer}\n")

    # Add a delay between questions to avoid overloading the server
    if i < len(questions) - 1:
        time.sleep(3)

# Combine all answers into a single report
final_report = "# Chicago Artwork Code Evaluation\n\n" + "\n".join(all_answers)

# Save the final report
with open("final_code_evaluation.md", "w", encoding="utf-8") as f:
    f.write(final_report)

print(f"\nEvaluation complete! Final report saved to final_code_evaluation.md")