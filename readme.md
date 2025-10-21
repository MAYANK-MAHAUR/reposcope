# RepoScopeAgent

**RepoScopeAgent** is an intelligent, LLM-driven tool built with the [Sentient Agent Framework](https://github.com/sentient-agi/Sentient-Agent-Framework) to analyze GitHub repositories. It simplifies the process of understanding, summarizing, and comparing open-source projects by providing a conversational interface for tasks like code analysis, issue detection, pattern searching, line counting, and version diffing. Designed for developers, it integrates seamlessly with sentient.xyz, offering a user-friendly way to explore repositories without manual cloning or complex tooling.


## Problem It Solves

The majority of users exploring tools like this are developers — most come looking for **coding-related insights**. But the process is often messy:

* **Limitations of Standard LLMs**: Traditional models like ChatGPT can only **crawl a repo’s front page** — they can’t **deep dive into the full codebase**. That means their answers often lack context, clarity, and accuracy when analyzing or summarizing repositories.
* **Time-Consuming Manual Analysis**: Cloning repositories, running linters, or comparing versions manually is tedious and error-prone.
* **Complex Code Insights**: Identifying code patterns, function calls, or potential issues requires specialized tools and expertise.
* **Version Comparison Challenges**: Understanding changes between repository versions (e.g., new features, bug fixes) involves complex Git commands.
* **Security and Reliability Concerns**: Assessing whether a repository is secure or trustworthy requires deep inspection of code and licenses.

**RepoScopeAgent** solves these by:
- **Automating Analysis**: Uses an LLM to classify queries and provide insights like summaries, security assessments, and code metrics.
- **Streamlining Version Diffs**: Compares repository versions (e.g., `v1.0.0` vs `v2.0.0`) with clear tables of added/removed lines.
- **Simplifying Code Inspection**: Finds patterns (e.g., `try-except`), counts lines, and detects issues using tools like pylint.
- **Ensuring Trustworthiness**: Analyzes licenses and code for malware, providing confidence in using open-source projects.
- **Conversational Interface**: Supports natural language queries (e.g., “Is pallets/flask secure?”) via a chat interface, ideal for sentient.xyz users.

## Features

- **Repository Analysis**: Summarizes repository contents, structure, and security (e.g., `is github.com/pallets/flask secure`).
- **Version Diffing**: Compares two versions of a repository to show file changes (e.g., `diff github.com/pallets/flask v1.0.0 v2.0.0`).
- **Code Pattern Search**: Finds specific patterns like `try-except` or function calls (e.g., `find try-except in pallets/flask`).
- **Code Issue Detection**: Identifies potential issues using pylint (e.g., `find issues in pallets/flask`).
- **Line and File Counting**: Counts lines of code and files by type (e.g., `how many lines of code in pallets/flask`).
- **Function Call Tracing**: Traces function calls across a repository (e.g., `trace route in pallets/flask`).
- **Comparison Across Repos**: Compares function implementations across multiple repositories (e.g., `compare route in pallets/flask and tiangolo/fastapi`).
- **Default Repo Setting**: Sets a default repository for repeated analysis (e.g., `set default github.com/pallets/flask`).
- **LLM-Driven Queries**: Supports natural language inputs for flexible, user-friendly interaction.
- **Clean Output**: Delivers tabulated, Markdown-formatted results for easy reading in a chat interface.

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