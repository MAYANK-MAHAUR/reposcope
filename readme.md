# RepoScopeAgent

**RepoScopeAgent** is an intelligent, LLM-driven tool built with the [Sentient Agent Framework](https://github.com/sentient-agi/Sentient-Agent-Framework) to streamline GitHub repository analysis. It empowers developers to explore, summarize, and compare open-source projects through a conversational chat interface, ideal for platforms like sentient.xyz. By automating tasks like code summarization, version diffing, pattern searching, and issue detection, it saves time and simplifies complex workflows.
Here’s a tightened-up version with your new points folded in naturally:


## Problem It Solves

The majority of users exploring tools like this are developers — most come looking for **coding-related insights**. But the process is often messy:

* **Limitations of Standard LLMs**: Traditional models like ChatGPT can only **crawl a repo’s front page** — they can’t **deep dive into the full codebase**. That means their answers often lack context, clarity, and accuracy when analyzing or summarizing repositories.
* **Time-Consuming Manual Analysis**: Cloning repositories, running linters, or comparing versions manually is tedious and error-prone.
* **Complex Code Insights**: Identifying code patterns, function calls, or potential issues requires specialized tools and expertise.
* **Version Comparison Challenges**: Understanding changes between repository versions (e.g., new features, bug fixes) involves complex Git commands.
* **Security and Reliability Concerns**: Assessing whether a repository is secure or trustworthy requires deep inspection of code and licenses.


**RepoScopeAgent** addresses these by:
- **Automating Analysis**: Uses an LLM (Qwen-72B) to interpret natural language queries and deliver insights like summaries, security assessments, and code metrics.
- **Simplifying Version Diffs**: Compares repository versions (e.g., `v1.0.0` vs `v2.0.0`) with clear tables showing added/removed lines.
- **Streamlining Code Inspection**: Finds patterns (e.g., `try-except`), counts lines, and detects issues using tools like pylint.
- **Ensuring Reliability**: Analyzes licenses and code for malware, providing confidence in using open-source projects.
- **Conversational Interface**: Supports flexible, natural language queries (e.g., “Is pallets/flask secure?”) for a seamless user experience on sentient.xyz.

## Features

- **Repository Analysis**: Summarizes repository contents, structure, and security (e.g., `is github.com/pallets/flask secure`).
- **Version Diffing**: Compares two versions of a repository to highlight file changes (e.g., `diff github.com/pallets/flask v1.0.0 v2.0.0`).
- **Code Pattern Search**: Identifies specific patterns like `try-except` or function calls (e.g., `find try-except in pallets/flask`).
- **Code Issue Detection**: Detects potential issues using pylint (e.g., `find issues in pallets/flask`).
- **Line and File Counting**: Counts lines of code and files by type (e.g., `how many lines of code in pallets/flask`).
- **Function Call Tracing**: Traces function calls across a repository (e.g., `trace route in pallets/flask`).
- **Cross-Repository Comparison**: Compares function implementations across multiple repositories (e.g., `compare route in pallets/flask and tiangolo/fastapi`).
- **Default Repository Setting**: Sets a default repository for repeated analysis (e.g., `set default github.com/pallets/flask`).
- **LLM-Driven Queries**: Handles natural language inputs for flexible, user-friendly interactions.
- **Clean, Tabulated Output**: Delivers Markdown-formatted tables for readability in chat interfaces.

## Installation

### Prerequisites
- Python 3.8+
- Git
- SQLite
- Required Python packages (see `requirements.txt`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/reposcopeagent.git
   cd reposcopeagent
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages:
   - `aiohttp`
   - `sentence-transformers`
   - `faiss-cpu`
   - `pygit2`
   - `pylint`

3. **Set Environment Variables**:
   ```bash
   export DEBUG_MODE=false  # Set to "true" for debug logs
   ```

4. **Run the Server**:
   ```bash
   python server.py
   ```
   The server runs on `http://127.0.0.1:8000/assist`.

5. **Test the Agent**:
   ```bash
   python test.py
   ```
   This executes a test suite and starts an interactive chat session.

### Requirements File
Create a `requirements.txt` with:
```
aiohttp
sentence-transformers
faiss-cpu
pygit2
pylint
```

## Usage

Run `test.py` to execute the test suite or interact via the chat interface.

### Example Commands
1. **Set Default Repository**:
   ```bash
   > set default github.com/pallets/flask
   ```
   **Output**:
   ```
   === Sending Prompt: set default github.com/pallets/flask ===
   --- 📝 Block Received: status ---
   Set default repo to pallets/flask.
   --- ✅ status COMPLETE ---
   ```

2. **Analyze Repository**:
   ```bash
   > is github.com/pallets/flask secure
   ```
   **Output**:
   ```
   === Sending Prompt: is github.com/pallets/flask secure ===
   --- 📝 Block Received: status ---
   Loading existing index for pallets/flask...
   --- 📝 Block Received: analysis ---
   The `pallets/flask` repository is reliable, secure, and free from malicious code.
   ### 1. License and Legal Clauses
   - Standard open-source license (LICENSE.txt).
   - No warranties, typical for open-source projects.
   ### 2. No Code Issues
   - No bugs, vulnerabilities, or malicious code found.
   ### 3. Authentication Mechanisms
   - Uses flask.session, @login_required, and supports Flask-Login.
   ### 4. No Signs of Malware
   - Only standard code and license text found.
   ### 5. Project Reputation
   - Well-maintained by Pallets with an active community.
   ### Conclusion
   Safe to use in your project.
   --- ✅ analysis COMPLETE ---
   ```

3. **Compare Versions**:
   ```bash
   > diff github.com/pallets/flask v1.0.0 v2.0.0
   ```
   **Output**:
   ```
   === Sending Prompt: diff github.com/pallets/flask v1.0.0 v2.0.0 ===
   --- 📝 Block Received: status ---
   Comparing pallets/flask versions v1.0.0 and v2.0.0...
   --- 📝 Block Received: analysis ---
   Diff Results for pallets/flask (v1.0.0 vs v2.0.0):
   | File | Lines Added | Lines Removed |
   |------|-------------|---------------|
   | src/flask/app.py | 150 | 120 |
   | src/flask/cli.py | 80 | 50 |
   | tests/test_basic.py | 200 | 180 |
   --- ✅ analysis COMPLETE ---
   ```

4. **Find Code Patterns**:
   ```bash
   > find try-except in pallets/flask
   ```
   **Output**:
   ```
   === Sending Prompt: find try-except in pallets/flask ===
   --- 📝 Block Received: status ---
   Searching for try-except patterns in pallets/flask...
   --- 📝 Block Received: analysis ---
   Try-Except Patterns Found:
   | Repo | File | Line | Pattern | Code |
   |------|------|------|---------|------|
   | pallets/flask | src/flask/app.py | 123 | try-except | try: ... except ValueError: ... |
   --- ✅ analysis COMPLETE ---
   ```

5. **Count Lines of Code**:
   ```bash
   > how many lines of code in pallets/flask
   ```
   **Output**:
   ```
   === Sending Prompt: how many lines of code in pallets/flask ===
   --- 📝 Block Received: status ---
   Counting lines of code for pallets/flask...
   --- 📝 Block Received: analysis ---
   Line Count Results:
   | Repo | Total Lines | Python Files | Python Lines | JS Files | JS Lines | Other Files | Other Lines |
   |------|-------------|--------------|--------------|----------|----------|-------------|-------------|
   | pallets/flask | 15000 | 50 | 12000 | 5 | 1000 | 10 | 2000 |
   --- ✅ analysis COMPLETE ---
   ```

### Interactive Chat
Run `python test.py` and enter prompts at the `>` prompt. Type `exit` or `quit` to end the session.

## Project Structure
- `agent.py`: Core logic for intent classification and repository analysis.
- `github_adapter.py`: Handles GitHub interactions (cloning, diffing, file fetching).
- `database.py`: Manages session and index storage using SQLite.
- `data_formatter.py`: Formats code chunks and interactions for LLM input.
- `server.py`: Runs the HTTP server for API requests.
- `test.py`: Client script for testing and interactive chat.
- `reposcope_sessions.db`: SQLite database for session data.
- `reposcope_indexes/`: Directory for FAISS indexes and chunk data.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License
MIT License. See `LICENSE` for details.

## Future Improvements
- **Branch Comparison**: Support diffing branches (e.g., `diff pallets/flask main dev`).
- **Pattern Diffing**: Identify changes in code patterns between versions.
- **Caching**: Store diff results in the database for performance optimization.
- **UI Integration**: Enhance output for sentient.xyz’s web interface (e.g., HTML tables).
- **Rate Limiting**: Implement GitHub API rate limiting for production use.

## Contact
For issues or feature requests, open a GitHub issue or contact the maintainers at sentient.xyz.