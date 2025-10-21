import aiosqlite
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MessageDatabase:
    def __init__(self, db_path: str = "reposcope_sessions.db"):
        self.db_path = db_path
       
        import asyncio
        asyncio.run(self.initialize())

    async def initialize(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DROP TABLE IF EXISTS messages")
                await db.execute("""
                    CREATE TABLE messages (
                        session_id TEXT,
                        prompt TEXT,
                        repo_data TEXT,
                        timestamp INTEGER
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS indexes (
                        repo TEXT PRIMARY KEY,
                        index_path TEXT,
                        chunk_count INTEGER,
                        timestamp INTEGER
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS function_calls (
                        repo TEXT,
                        file_path TEXT,
                        caller TEXT,
                        callee TEXT,
                        line INTEGER,
                        line_content TEXT,
                        timestamp INTEGER
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS default_repo (
                        session_id TEXT PRIMARY KEY,
                        repo TEXT,
                        timestamp INTEGER
                    )
                """)
                await db.commit()
                logger.debug("Database initialized successfully with updated schema")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def store_message(self, session_id: str, prompt: str, repo_data: Optional[Dict[str, Any]], timestamp: int):
        try:
            repo_data_json = json.dumps(repo_data) if repo_data else None
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (session_id, prompt, repo_data, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, prompt, repo_data_json, timestamp)
                )
                await db.commit()
                logger.debug(f"Stored message for session {session_id}: {prompt}")
        except Exception as e:
            logger.error(f"Failed to store message: {str(e)}")
            raise

    async def retrieve_messages(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT prompt, repo_data, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,)
                )
                rows = await cursor.fetchall()
                messages = []
                for row in rows:
                    prompt, repo_data_json, timestamp = row
                    repo_data = json.loads(repo_data_json) if repo_data_json else None
                    messages.append({"prompt": prompt, "repo_data": repo_data, "timestamp": timestamp})
                await cursor.close()
                return messages
        except Exception as e:
            logger.error(f"Failed to retrieve messages: {str(e)}")
            return []

    async def store_index(self, repo: str, index_path: str, chunk_count: int, timestamp: int):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO indexes (repo, index_path, chunk_count, timestamp) VALUES (?, ?, ?, ?)",
                    (repo, index_path, chunk_count, timestamp)
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to store index: {str(e)}")

    async def retrieve_index(self, repo: str) -> Optional[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT index_path, chunk_count, timestamp FROM indexes WHERE repo = ?",
                    (repo,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                    return {"index_path": row[0], "chunk_count": row[1], "timestamp": row[2]}
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve index: {str(e)}")
            return None

    async def store_function_calls(self, repo: str, calls: List[Dict[str, Any]], timestamp: int):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for call in calls:
                    await db.execute(
                        "INSERT INTO function_calls (repo, file_path, caller, callee, line, line_content, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (repo, call["file_path"], call["caller"], call["callee"], call["line"], call["line_content"], timestamp)
                    )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to store function calls: {str(e)}")

    async def retrieve_function_calls(self, repo: str, function_name: str = None) -> List[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if function_name:
                    cursor = await db.execute(
                        "SELECT file_path, caller, callee, line, line_content FROM function_calls WHERE repo = ? AND callee = ?",
                        (repo, function_name)
                    )
                else:
                    cursor = await db.execute(
                        "SELECT file_path, caller, callee, line, line_content FROM function_calls WHERE repo = ?",
                        (repo,)
                    )
                rows = await cursor.fetchall()
                calls = [
                    {"file_path": row[0], "caller": row[1], "callee": row[2], "line": row[3], "line_content": row[4]}
                    for row in rows
                ]
                await cursor.close()
                return calls
        except Exception as e:
            logger.error(f"Failed to retrieve function calls: {str(e)}")
            return []

    async def cleanup_session(self, session_id: str):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                await db.execute("DELETE FROM default_repo WHERE session_id = ?", (session_id,))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {str(e)}")

    async def cleanup_index(self, repo: str):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM indexes WHERE repo = ?", (repo,))
                await db.execute("DELETE FROM function_calls WHERE repo = ?", (repo,))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup index for {repo}: {str(e)}")

    async def set_default_repo(self, session_id: str, repo: str, timestamp: int):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO default_repo (session_id, repo, timestamp) VALUES (?, ?, ?)",
                    (session_id, repo, timestamp)
                )
                await db.commit()
                logger.debug(f"Set default repo {repo} for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to set default repo: {str(e)}")

    async def get_default_repo(self, session_id: str) -> Optional[str]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT repo FROM default_repo WHERE session_id = ?",
                    (session_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get default repo: {str(e)}")
            return None