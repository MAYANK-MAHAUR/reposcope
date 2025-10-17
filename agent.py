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
from github_adapter import clone_repo, fetch_repo_files, parse_function_calls
import time
import aiosqlite

logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler("reposcope.log"),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

message_db = MessageDatabase()

class RepoScopeAgent(AbstractAgent):
    def __init__(self):
        super().__init__(name="RepoScope")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.session_timeout = 300
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
                    (current_time - 86400,)
                )
                rows = await cursor.fetchall()
                for row in rows:
                    repo = row[0]
                    await message_db.cleanup_index(repo)
                await cursor.close()
            await asyncio.sleep(60)

    async def assist(self, session: Session, query: Query, response_handler: ResponseHandler):
        if self.cleanup_task is None:
            self.start_cleanup_task()

        activity_id_str = str(session.activity_id)
        current_time = int(time.time())
        self.active_sessions[activity_id_str] = current_time

        try:
            past_interactions = await message_db.retrieve_messages(activity_id_str)
            logger.debug(f"Retrieved {len(past_interactions)} interactions for session {activity_id_str}")

            repo_url_match = re.search(r'github\.com/([\w-]+)/([\w-]+)', query.prompt)
            trace_match = re.search(r'\btrace\s+([^\s]+)\b', query.prompt, re.IGNORECASE)
            is_new_repo = repo_url_match is not None
            is_trace_query = trace_match is not None
            current_repo = None
            index = None
            chunks = []

            if is_new_repo:
                user, repo = repo_url_match.groups()
                current_repo = f"{user}/{repo}"
                index_path = os.path.join(self.index_dir, f"{current_repo.replace('/', '_')}.faiss")
                chunk_path = index_path.replace(".faiss", ".json")
                repo_path = os.path.join(tempfile.gettempdir(), f"{user}_{repo}")

                # Parse function calls
                function_calls = await parse_function_calls(repo_path)
                await message_db.store_function_calls(current_repo, function_calls, current_time)

                # Check for existing index
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
                    logger.debug(f"No existing index for {current_repo}, cloning and indexing...")
                    await response_handler.emit_text_block("status", f"Cloning {current_repo}...")
                    await clone_repo(f"https://github.com/{user}/{repo}.git", repo_path)
                    files = await fetch_repo_files(repo_path)

                    # Chunk files
                    for file in files:
                        content = file["content"]
                        chunked = [content[i:i+500] for i in range(0, len(content), 500)]
                        chunks.extend([{"file_path": file["file_path"], "content": chunk} for chunk in chunked])

                    # Embed and index
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
                        return

                    await response_handler.emit_text_block("status", f"Indexed {len(chunks)} chunks from {current_repo}.")
                    await message_db.store_message(activity_id_str, f"Indexed repo: {current_repo}", {"repo": current_repo, "chunks": len(chunks)}, current_time)
            else:
                last_repo_msg = next((msg for msg in reversed(past_interactions) if msg["prompt"].startswith("Indexed repo")), None)
                if last_repo_msg:
                    current_repo = last_repo_msg["repo_data"]["repo"]
                    index_path = os.path.join(self.index_dir, f"{current_repo.replace('/', '_')}.faiss")
                    chunk_path = index_path.replace(".faiss", ".json")
                    repo_path = os.path.join(tempfile.gettempdir(), f"{current_repo.replace('/', '_')}")
                    if os.path.exists(index_path) and os.path.exists(chunk_path):
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
                        await response_handler.emit_text_block("status", f"Re-indexing {current_repo}...")
                        if not os.path.exists(repo_path):
                            user, repo = current_repo.split('/')
                            await clone_repo(f"https://github.com/{user}/{repo}.git", repo_path)
                        files = await fetch_repo_files(repo_path)
                        # Parse function calls
                        function_calls = await parse_function_calls(repo_path)
                        await message_db.store_function_calls(current_repo, function_calls, current_time)
                        for file in files:
                            content = file["content"]
                            chunked = [content[i:i+500] for i in range(0, len(content), 500)]
                            chunks.extend([{"file_path": file["file_path"], "content": chunk} for chunk in chunked])
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
                            return
                else:
                    await response_handler.emit_error("No repo indexed in this session. Please provide a GitHub link.", 400)
                    await response_handler.complete()
                    return

            if not index or not chunks:
                await response_handler.emit_error("No index available. Please provide a GitHub repo link.", 400)
                await response_handler.complete()
                return

            if is_trace_query:
                function_name = trace_match.group(1)
                await response_handler.emit_text_block("status", f"Tracing function {function_name} in {current_repo}...")
                function_calls = await message_db.retrieve_function_calls(current_repo, function_name)
                if not function_calls:
                    await response_handler.emit_text_block("analysis", f"No calls found for {function_name} in {current_repo}.")
                    await response_handler.complete()
                    return

                # Format call graph as text
                call_graph = []
                for call in function_calls:
                    caller = call["caller"] or "Unknown"
                    call_graph.append(
                        f"{caller} -> {call['callee']} at {call['file_path']}:{call['line']} ({call['line_content']})"
                    )
                call_graph_text = "\n".join(call_graph)

                # Optional: Mermaid diagram
                mermaid = ["graph TD"]
                for call in function_calls:
                    caller = call["caller"] or "Unknown"
                    mermaid.append(f"    {caller.replace('.', '_')} --> {call['callee'].replace('.', '_')}")
                mermaid_text = "\n".join(mermaid)

                await response_handler.emit_text_block("analysis", f"Call graph for {function_name}:\n{call_graph_text}\n\nMermaid diagram:\n```mermaid\n{mermaid_text}\n```")
                await message_db.store_message(activity_id_str, query.prompt, {"repo": current_repo, "query": query.prompt, "response": call_graph_text}, current_time)
                await response_handler.complete()
                return

            # Standard query: Search index
            query_emb = self.embedder.encode([query.prompt], show_progress_bar=False)
            query_emb = np.array(query_emb).astype('float32')
            _, indices = index.search(query_emb, k=5)
            relevant_chunks = [chunks[i] for i in indices[0]]
            formatted_chunks = format_all_chunks_for_llm(relevant_chunks)

            # Format history
            history_string = format_interactions_for_llm(past_interactions)

            # LLM prompt
            llm_prompt = (
                f"Repo: {current_repo}\n"
                f"Relevant code chunks:\n{formatted_chunks}\n\n"
                f"Conversation history:\n{history_string}\n\n"
                f"User Query: {query.prompt}\n\n"
                "Provide a clear, detailed analysis or summary based on the repo contents. Be explanatory and limit to key insights."
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
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        buffer += content
                        if any(buffer.endswith(p) for p in [".", "!", "?"]) or len(buffer) > 100:
                            await stream.emit_chunk(buffer)
                            buffer = ""
                if buffer:
                    await stream.emit_chunk(buffer)
                await stream.complete()
            except Exception as e:
                logger.error(f"GaiaNet LLM failed: {str(e)}")
                await response_handler.emit_error(f"Failed to generate analysis: {str(e)}", 500)

            # Store interaction
            repo_data = {
                "repo": current_repo,
                "query": query.prompt,
                "chunks_returned": len(relevant_chunks),
                "response": buffer
            }
            await message_db.store_message(activity_id_str, query.prompt, repo_data, current_time)

            await response_handler.complete()

            # Clean up temporary repo directory
            if 'repo_path' in locals() and repo_path and os.path.exists(repo_path):
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