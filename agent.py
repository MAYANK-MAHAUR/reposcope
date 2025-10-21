
import re
import logging
import asyncio
import numpy as np
import faiss
import json
import os
import shutil
import tempfile
from typing import List, Dict, Any
from sentient_agent_framework.interface.agent import AbstractAgent
from sentient_agent_framework.interface.session import Session
from sentient_agent_framework.interface.request import Query
from sentient_agent_framework.interface.response_handler import ResponseHandler
from sentence_transformers import SentenceTransformer
from config import async_client
from database import MessageDatabase
from data_formatter import format_all_chunks_for_llm, format_interactions_for_llm
from github_adapter import clone_repo, fetch_repo_files, parse_function_calls, count_lines_of_code, find_code_patterns, run_pylint
import time
import aiosqlite

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("reposcope.log"),
        logging.StreamHandler() if os.getenv("DEBUG_MODE", "false").lower() == "true" else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

message_db = MessageDatabase()

class RepoScopeAgent(AbstractAgent):
    def __init__(self):
        super().__init__(name="RepoScope")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.session_timeout = 3600
        self.active_sessions = {}
        self.index_dir = "reposcope_indexes"
        os.makedirs(self.index_dir, exist_ok=True)
        self.cleanup_task = None
        try:
            faiss_index = faiss.IndexFlatL2(384)
            logger.debug("FAISS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            raise ImportError("FAISS initialization failed. Ensure faiss-cpu is installed.")

    def start_cleanup_task(self):
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_stale_sessions())

    async def _cleanup_stale_sessions(self):
        logger.info("Cleanup task started - checking every 60 seconds")
        while True:
            current_time = int(time.time())
            stale_sessions = [
                sid for sid, last_active in self.active_sessions.items()
                if current_time - last_active > self.session_timeout
            ]
            for sid in stale_sessions:
                try:
                    await message_db.cleanup_session(sid)
                    logger.info(f"Cleaned up stale session: {sid}")
                    del self.active_sessions[sid]
                except Exception as e:
                    logger.error(f"Failed to clean up session {sid}: {str(e)}")
            async with aiosqlite.connect("reposcope_sessions.db") as db:
                cursor = await db.execute(
                    "SELECT repo FROM indexes WHERE timestamp < ?",
                    (current_time - 86400 * 7,)
                )
                rows = await cursor.fetchall()
                for row in rows:
                    repo = row[0]
                    await message_db.cleanup_index(repo)
                await cursor.close()
            await asyncio.sleep(60)

    async def classify_intent(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to classify the query intent and extract parameters."""
        llm_prompt = (
            f"Classify the following user query and extract parameters. Possible intents: analyze, trace, compare, lines_of_code, file_count, find_issues, find_pattern, set_default, or unknown.\n\n"
            f"For 'trace': extract function_name.\n"
            f"For 'compare': extract function_name and repos (list of user/repo, e.g., ['pallets/flask']).\n"
            f"For 'lines_of_code' or 'file_count': extract repos if specified (e.g., ['pallets/flask']), else use default.\n"
            f"For 'find_issues': extract repos if specified (e.g., ['pallets/flask']), else use default.\n"
            f"For 'find_pattern': extract pattern (e.g., 'try-except') and repos if specified (e.g., ['pallets/flask']), else use default.\n"
            f"For 'set_default': extract repo (e.g., 'pallets/flask').\n"
            f"For 'analyze': general analysis, extract repos if specified (e.g., ['pallets/flask']), else use default.\n"
            f"Return JSON: {{ \"intent\": \"intent_name\", \"function_name\": \"str or null\", \"pattern\": \"str or null\", \"repos\": [\"list of user/repo\"] or null }}\n\n"
            f"Examples:\n"
            f"'set default github.com/pallets/flask' -> {{'intent': 'set_default', 'repos': ['pallets/flask'], 'function_name': null, 'pattern': null}}\n"
            f"'find try-except' -> {{'intent': 'find_pattern', 'pattern': 'try-except', 'repos': null, 'function_name': null}}\n"
            f"'summarize github.com/pallets/flask' -> {{'intent': 'analyze', 'repos': ['pallets/flask'], 'function_name': null, 'pattern': null}}\n"
            f"'is github.com/pallets/flask secure' -> {{'intent': 'analyze', 'repos': ['pallets/flask'], 'function_name': null, 'pattern': null}}\n"
            f"Query: {prompt}"
        )
        try:
            response = await async_client.chat.completions.create(
                model="qwen-72b",
                messages=[{"role": "user", "content": llm_prompt}],
                max_tokens=200
            )
            intent_data = json.loads(response.choices[0].message.content)
            logger.debug(f"Classified intent: {intent_data}")
            return intent_data
        except Exception as e:
            logger.error(f"Failed to classify intent: {str(e)}")
            return {"intent": "unknown", "repos": None, "function_name": None, "pattern": None}

    async def assist(self, session: Session, query: Query, response_handler: ResponseHandler):
        if self.cleanup_task is None:
            self.start_cleanup_task()

        activity_id_str = str(session.activity_id)
        current_time = int(time.time())
        self.active_sessions[activity_id_str] = current_time

        try:
            past_interactions = await message_db.retrieve_messages(activity_id_str)
            logger.debug(f"Retrieved {len(past_interactions)} interactions for session {activity_id_str}")

            # Classify intent using LLM
            intent_data = await self.classify_intent(query.prompt)
            intent = intent_data["intent"]
            function_name = intent_data.get("function_name")
            pattern = intent_data.get("pattern")
            specified_repos = intent_data.get("repos")

            repos_data = {}

            # Normalize and validate repo string
            def normalize_repo(repo: str) -> tuple[str, str]:
                """Convert repo string (e.g., 'github.com/user/repo' or 'user/repo') to (user, repo)."""
                repo = repo.strip('/')
                if repo.startswith('github.com/'):
                    parts = repo.split('/')
                    if len(parts) >= 3:
                        return parts[-2], parts[-1]
                    else:
                        raise ValueError(f"Invalid repo format: {repo}. Expected 'github.com/user/repo' or 'user/repo'.")
                parts = repo.split('/')
                if len(parts) == 2:
                    return parts[0], parts[1]
                raise ValueError(f"Invalid repo format: {repo}. Expected 'github.com/user/repo' or 'user/repo'.")

            # Load or index repos
            async def load_or_index_repo(user: str, repo: str, repo_path: str, index_path: str, chunk_path: str):
                current_repo = f"{user}/{repo}"
                index_metadata = await message_db.retrieve_index(current_repo)
                if index_metadata and os.path.exists(index_path) and os.path.exists(chunk_path):
                    await response_handler.emit_text_block("status", f"Loading existing index for {current_repo}...")
                    try:
                        index = faiss.read_index(index_path)
                        with open(chunk_path, 'r') as f:
                            chunks = json.load(f)
                        logger.debug(f"Loaded {len(chunks)} chunks from {index_path}")
                    except Exception as e:
                        logger.error(f"Failed to load index for {current_repo}: {str(e)}")
                        index = None
                        chunks = []
                else:
                    await response_handler.emit_text_block("status", f"Cloning {current_repo}...")
                    await clone_repo(f"https://github.com/{user}/{repo}.git", repo_path)
                    files = await fetch_repo_files(repo_path)
                    function_calls = await parse_function_calls(repo_path, current_repo)
                    await message_db.store_function_calls(current_repo, function_calls, current_time)

                    chunks = []
                    for file in files:
                        content = file["content"]
                        chunked = [content[i:i+500] for i in range(0, len(content), 500)]
                        chunks.extend([{"file_path": file["file_path"], "content": chunk, "repo": current_repo} for chunk in chunked])

                    chunk_texts = [chunk["content"] for chunk in chunks]
                    embeddings = self.embedder.encode(chunk_texts, show_progress_bar=False)
                    embeddings = np.array(embeddings).astype('float32')
                    try:
                        dimension = embeddings.shape[1]
                        index = faiss.IndexFlatL2(dimension)
                        index.add(embeddings)
                        faiss.write_index(index, index_path)
                        with open(chunk_path, 'w') as f:
                            json.dump(chunks, f)
                        await message_db.store_index(current_repo, index_path, len(chunks), current_time)
                        logger.debug(f"Saved index for {current_repo} to {index_path}")
                    except Exception as e:
                        logger.error(f"FAISS indexing failed: {str(e)}")
                        await response_handler.emit_error(f"Failed to index repo: {str(e)}", 500)
                        await response_handler.complete()
                        return None
                    await response_handler.emit_text_block("status", f"Indexed {len(chunks)} chunks from {current_repo}.")
                return current_repo, {"index": index, "chunks": chunks, "repo_path": repo_path}

            if specified_repos:
                for repo in specified_repos:
                    try:
                        user, repo_name = normalize_repo(repo)
                        repo_path = os.path.join(tempfile.gettempdir(), f"{user}_{repo_name}")
                        index_path = os.path.join(self.index_dir, f"{user}_{repo_name}.faiss")
                        chunk_path = index_path.replace(".faiss", ".json")
                        result = await load_or_index_repo(user, repo_name, repo_path, index_path, chunk_path)
                        if result:
                            repos_data[result[0]] = result[1]
                    except ValueError as e:
                        logger.error(f"Invalid repo format: {str(e)}")
                        await response_handler.emit_text_block("error", str(e))
                        await response_handler.complete()
                        return
            else:
                default_repo = await message_db.get_default_repo(activity_id_str)
                if default_repo:
                    try:
                        user, repo = normalize_repo(default_repo)
                        repo_path = os.path.join(tempfile.gettempdir(), f"{user}_{repo}")
                        index_path = os.path.join(self.index_dir, f"{user}_{repo}.faiss")
                        chunk_path = index_path.replace(".faiss", ".json")
                        result = await load_or_index_repo(user, repo, repo_path, index_path, chunk_path)
                        if result:
                            repos_data[result[0]] = result[1]
                    except ValueError as e:
                        logger.error(f"Invalid default repo format: {str(e)}")
                        await response_handler.emit_text_block("error", f"Invalid default repo: {str(e)}. Please set a valid default repo.")
                        await response_handler.complete()
                        return
                else:
                    last_repos = [msg["repo_data"]["repo"] for msg in past_interactions if msg["repo_data"] and "repo" in msg["repo_data"]]
                    if last_repos:
                        for repo in set(last_repos[-2:]):
                            try:
                                user, repo_name = normalize_repo(repo)
                                repo_path = os.path.join(tempfile.gettempdir(), f"{user}_{repo_name}")
                                index_path = os.path.join(self.index_dir, f"{user}_{repo_name}.faiss")
                                chunk_path = index_path.replace(".faiss", ".json")
                                result = await load_or_index_repo(user, repo_name, repo_path, index_path, chunk_path)
                                if result:
                                    repos_data[result[0]] = result[1]
                            except ValueError as e:
                                logger.error(f"Invalid repo format in history: {str(e)}")
                                continue
                if not repos_data and intent != "unknown":
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'summarize github.com/pallets/flask') or set a default (e.g., 'set default github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return

            if intent == "set_default":
                if specified_repos:
                    try:
                        user, repo_name = normalize_repo(specified_repos[0])
                        default_repo = f"{user}/{repo_name}"
                        await message_db.set_default_repo(activity_id_str, default_repo, current_time)
                        await response_handler.emit_text_block("status", f"Set default repo to {default_repo}.")
                    except ValueError as e:
                        logger.error(f"Invalid repo format: {str(e)}")
                        await response_handler.emit_text_block("error", str(e))
                    await response_handler.complete()
                    return
                else:
                    await response_handler.emit_text_block("status", "No repo specified for default. Please provide a GitHub link (e.g., 'set default github.com/pallets/flask').")
                    await response_handler.complete()
                    return

            if intent == "trace" and function_name:
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'trace route in github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Tracing function {function_name} in {', '.join(repos_data.keys())}...")
                call_graph_text = ""
                mermaid = ["graph TD"]
                for repo, data in repos_data.items():
                    function_calls = await message_db.retrieve_function_calls(repo, function_name)
                    if function_calls:
                        call_graph_text += f"\n{repo}:\n"
                        for call in function_calls:
                            caller = call["caller"] or "Unknown"
                            line = f"{caller} -> {call['callee']} at {call['file_path']}:{call['line']} ({call['line_content']})\n"
                            call_graph_text += line
                            mermaid.append(f"    {caller.replace('.', '_')}_{repo.replace('/', '_')} --> {call['callee'].replace('.', '_')}_{repo.replace('/', '_')}")
                if not call_graph_text:
                    await response_handler.emit_text_block("analysis", f"No calls found for {function_name}.")
                    await response_handler.complete()
                    return

                mermaid_text = "\n".join(mermaid)
                analysis_text = f"Call graph for {function_name}:\n{call_graph_text}\nMermaid diagram:\n```mermaid\n{mermaid_text}\n```"
                await response_handler.emit_text_block("analysis", analysis_text)
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": analysis_text}, current_time)
                await response_handler.complete()
                return

            if intent == "compare" and function_name:
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please provide GitHub links (e.g., 'compare route in github.com/pallets/flask and github.com/tiangolo/fastapi')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Comparing {', '.join(repos_data.keys())}...")
                comparison_results = []
                for repo, data in repos_data.items():
                    function_calls = await message_db.retrieve_function_calls(repo, function_name)
                    if function_calls:
                        comparison_results.append({
                            "repo": repo,
                            "calls": [
                                {
                                    "caller": call["caller"] or "Unknown",
                                    "callee": call["callee"],
                                    "file_path": call["file_path"],
                                    "line": call["line"]
                                }
                                for call in function_calls
                            ]
                        })
                    else:
                        query_emb = self.embedder.encode([query.prompt], show_progress_bar=False)
                        query_emb = np.array(query_emb).astype('float32')
                        _, indices = data["index"].search(query_emb, k=5)
                        relevant_chunks = [data["chunks"][i] for i in indices[0]]
                        comparison_results.append({
                            "repo": repo,
                            "calls": [f"Chunk from {chunk['file_path']}: {chunk['content'][:100]}..." for chunk in relevant_chunks]
                        })

                table_lines = ["| Repo | Caller | Callee | File | Line |", "|------|--------|--------|------|------|"]
                for result in comparison_results:
                    repo = result["repo"]
                    if isinstance(result["calls"][0], dict):
                        for call in result["calls"]:
                            table_lines.append(
                                f"| {repo} | {call['caller']} | {call['callee']} | {call['file_path']} | {call['line']} |"
                            )
                    else:
                        for chunk in result["calls"]:
                            table_lines.append(f"| {repo} | - | - | {chunk.split(': ')[0].split(' ')[2]} | - |")
                comparison_text = "Comparison Results:\n" + "\n".join(table_lines) + "\n"

                mermaid = ["graph TD"]
                for result in comparison_results:
                    if isinstance(result["calls"][0], dict):
                        for call in result["calls"]:
                            caller = call["caller"].replace('.', '_')
                            callee = call["callee"].replace('.', '_')
                            repo_id = result["repo"].replace('/', '_')
                            mermaid.append(f"    {caller}_{repo_id} --> {callee}_{repo_id}")
                mermaid_text = "\n".join(mermaid)

                await response_handler.emit_text_block("analysis", f"{comparison_text}\nMermaid diagram:\n```mermaid\n{mermaid_text}\n```")
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": comparison_text}, current_time)
                await response_handler.complete()
                return

            if intent == "lines_of_code":
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'how many lines of code in github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Counting lines of code for {', '.join(repos_data.keys())}...")
                line_counts = []
                for repo, data in repos_data.items():
                    line_count = await count_lines_of_code(data["repo_path"])
                    line_counts.append({"repo": repo, **line_count})

                table_lines = [
                    "| Repo | Total Lines | Python Files | Python Lines | JS Files | JS Lines | Other Files | Other Lines |",
                    "|------|-------------|--------------|--------------|----------|----------|-------------|-------------|"
                ]
                for count in line_counts:
                    table_lines.append(
                        f"| {count['repo']} | {count['total_lines']} | {count['python_files']} | {count['python_lines']} | "
                        f"{count['js_files']} | {count['js_lines']} | {count['other_files']} | {count['other_lines']} |"
                    )
                table_text = "\n".join(table_lines)
                await response_handler.emit_text_block("analysis", f"Line Count Results:\n{table_text}")
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": table_text}, current_time)
                await response_handler.complete()
                return

            if intent == "file_count":
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'file count in github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Counting files for {', '.join(repos_data.keys())}...")
                file_counts = []
                for repo, data in repos_data.items():
                    files = await fetch_repo_files(data["repo_path"])
                    python_files = sum(1 for f in files if f["file_path"].endswith('.py'))
                    js_files = sum(1 for f in files if f["file_path"].endswith('.js'))
                    other_files = len(files) - python_files - js_files
                    file_counts.append({
                        "repo": repo,
                        "total_files": len(files),
                        "python_files": python_files,
                        "js_files": js_files,
                        "other_files": other_files
                    })

                table_lines = [
                    "| Repo | Total Files | Python Files | JS Files | Other Files |",
                    "|------|-------------|--------------|----------|-------------|"
                ]
                for count in file_counts:
                    table_lines.append(
                        f"| {count['repo']} | {count['total_files']} | {count['python_files']} | {count['js_files']} | {count['other_files']} |"
                    )
                table_text = "\n".join(table_lines)
                await response_handler.emit_text_block("analysis", f"File Count Results:\n{table_text}")
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": table_text}, current_time)
                await response_handler.complete()
                return

            if intent == "find_issues":
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'find issues in github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Scanning for code issues in {', '.join(repos_data.keys())}...")
                issue_results = []
                for repo, data in repos_data.items():
                    issues = await run_pylint(data["repo_path"], repo)
                    issue_results.append({"repo": repo, "issues": issues})

                table_lines = ["| Repo | File | Line | Issue Code | Message |", "|------|------|------|------------|---------|"]
                for result in issue_results:
                    for issue in result["issues"]:
                        table_lines.append(
                            f"| {result['repo']} | {issue['file_path']} | {issue['line']} | {issue['issue_code']} | {issue['message']} |"
                        )
                issues_text = "Code Issues Found:\n" + "\n".join(table_lines) if table_lines[2:] else "No code issues found."
                await response_handler.emit_text_block("analysis", issues_text)
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": issues_text}, current_time)
                await response_handler.complete()
                return

            if intent == "find_pattern" and pattern:
                if not repos_data:
                    await response_handler.emit_text_block(
                        "status",
                        "No repos indexed or default set. Please specify a repo (e.g., 'find try-except in github.com/pallets/flask')."
                    )
                    await response_handler.complete()
                    return
                await response_handler.emit_text_block("status", f"Searching for {pattern} patterns in {', '.join(repos_data.keys())}...")
                pattern_results = []
                for repo, data in repos_data.items():
                    patterns = await find_code_patterns(data["repo_path"], pattern, repo)
                    pattern_results.append({"repo": repo, "patterns": patterns})

                table_lines = ["| Repo | File | Line | Pattern | Code |", "|------|------|------|---------|------|"]
                for result in pattern_results:
                    for pattern in result["patterns"]:
                        table_lines.append(
                            f"| {result['repo']} | {pattern['file_path']} | {pattern['line']} | {pattern['pattern']} | {pattern['line_content'][:50]}... |"
                        )
                patterns_text = f"{pattern.capitalize()} Patterns Found:\n" + "\n".join(table_lines) if table_lines[2:] else f"No {pattern} patterns found."
                await response_handler.emit_text_block("analysis", patterns_text)
                await message_db.store_message(activity_id_str, query.prompt, {"repos": list(repos_data.keys()), "query": query.prompt, "response": patterns_text}, current_time)
                await response_handler.complete()
                return

            if not repos_data:
                await response_handler.emit_text_block(
                    "status",
                    "No repos indexed or default set. Please specify a repo (e.g., 'summarize github.com/pallets/flask') or set a default (e.g., 'set default github.com/pallets/flask')."
                )
                await response_handler.complete()
                return

            current_repo = next(iter(repos_data))
            index = repos_data[current_repo]["index"]
            chunks = repos_data[current_repo]["chunks"]

            query_emb = self.embedder.encode([query.prompt], show_progress_bar=False)
            query_emb = np.array(query_emb).astype('float32')
            _, indices = index.search(query_emb, k=5)
            relevant_chunks = [chunks[i] for i in indices[0]]
            formatted_chunks = format_all_chunks_for_llm(relevant_chunks)

            history_string = format_interactions_for_llm(past_interactions)

            llm_prompt = (
                f"Repo: {current_repo}\n"
                f"Relevant code chunks:\n{formatted_chunks}\n\n"
                f"Conversation history:\n{history_string}\n\n"
                f"User Query: {query.prompt}\n\n"
                "Provide a clear, detailed analysis or summary based on the repo contents in a single cohesive response. "
                "Use Markdown formatting with headings and lists for clarity. Limit to key insights."
            )
            try:
                response = await async_client.chat.completions.create(
                    model="qwen-72b",
                    messages=[{"role": "user", "content": llm_prompt}],
                    max_tokens=600,
                    stream=True
                )
                stream = response_handler.create_text_stream("analysis")
                buffer = ""
                paragraph_threshold = 500 
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        buffer += content
                        if "\n\n" in buffer or len(buffer) > paragraph_threshold:
                            parts = buffer.split("\n\n")
                            for part in parts[:-1]: 
                                if part.strip() and len(part.strip()) > 1: 
                                    await stream.emit_chunk(part.strip())
                            buffer = parts[-1]  
                if buffer.strip() and len(buffer.strip()) > 1: 
                    await stream.emit_chunk(buffer.strip())
                await stream.complete()
            except Exception as e:
                logger.error(f"GaiaNet LLM failed: {str(e)}")
                await response_handler.emit_error(f"Failed to generate analysis: {str(e)}", 500)

            repo_data = {
                "repo": current_repo,
                "query": query.prompt,
                "chunks_returned": len(relevant_chunks),
                "response": buffer
            }
            await message_db.store_message(activity_id_str, query.prompt, repo_data, current_time)

            await response_handler.complete()

            for repo, data in repos_data.items():
                repo_path = data["repo_path"]
                if repo_path and os.path.exists(repo_path):
                    shutil.rmtree(repo_path, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary repo directory: {repo_path}")

        except Exception as e:
            logger.error(f"RepoScope Agent failed: {str(e)}", exc_info=True)
            await response_handler.emit_error(
                error_message=f"RepoScope Agent failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )
            await message_db.store_message(activity_id_str, query.prompt, None, current_time)
            await response_handler.complete()
