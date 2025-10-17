import aiosqlite
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MessageDatabase:
    def __init__(self, db_path: str = "reposcope_sessions.db"):
        self.db_path = db_path
        self.initialized = False

    async def _init_db(self):
        """Initialize the database with messages, indexes, and function_calls tables."""
        if self.initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            # Messages table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activity_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    repo_data TEXT,
                    timestamp INTEGER NOT NULL
                )
            """)
            # Indexes table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS indexes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL UNIQUE,
                    index_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            # Function calls table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS function_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    caller TEXT,
                    callee TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    line_content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            await db.commit()
            self.initialized = True
            logger.debug("Initialized messages, indexes, and function_calls tables in database")

    async def store_message(self, activity_id: str, prompt: str, repo_data: Dict[str, Any] | None, timestamp: int) -> None:
        """Store a message and repo metadata in the database."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                repo_data_json = json.dumps(repo_data) if repo_data else None
                await db.execute(
                    "INSERT INTO messages (activity_id, prompt, repo_data, timestamp) VALUES (?, ?, ?, ?)",
                    (activity_id, prompt, repo_data_json, timestamp)
                )
                await db.commit()
                logger.debug(f"Stored message for activity_id {activity_id}")
            except Exception as e:
                logger.error(f"Failed to store message for activity_id {activity_id}: {str(e)}")
                raise

    async def store_index(self, repo: str, index_path: str, chunk_count: int, timestamp: int) -> None:
        """Store index metadata for a repo."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    "INSERT OR REPLACE INTO indexes (repo, index_path, chunk_count, timestamp) VALUES (?, ?, ?, ?)",
                    (repo, index_path, chunk_count, timestamp)
                )
                await db.commit()
                logger.debug(f"Stored index metadata for repo {repo}")
            except Exception as e:
                logger.error(f"Failed to store index for repo {repo}: {str(e)}")
                raise

    async def store_function_calls(self, repo: str, function_calls: List[Dict[str, Any]], timestamp: int) -> None:
        """Store function call metadata for a repo."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                for call in function_calls:
                    await db.execute(
                        "INSERT INTO function_calls (repo, file_path, caller, callee, line, line_content, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (repo, call["file_path"], call["caller"], call["callee"], call["line"], call["line_content"], timestamp)
                    )
                await db.commit()
                logger.debug(f"Stored {len(function_calls)} function calls for repo {repo}")
            except Exception as e:
                logger.error(f"Failed to store function calls for repo {repo}: {str(e)}")
                raise

    async def retrieve_index(self, repo: str) -> Dict[str, Any] | None:
        """Retrieve index metadata for a repo."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                cursor = await db.execute(
                    "SELECT index_path, chunk_count, timestamp FROM indexes WHERE repo = ?",
                    (repo,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                    logger.debug(f"Retrieved index metadata for repo {repo}")
                    return {"index_path": row[0], "chunk_count": row[1], "timestamp": row[2]}
                logger.debug(f"No index found for repo {repo}")
                return None
            except Exception as e:
                logger.error(f"Failed to retrieve index for repo {repo}: {str(e)}")
                raise

    async def retrieve_function_calls(self, repo: str, function_name: str) -> List[Dict[str, Any]]:
        """Retrieve function calls for a given repo and function name."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                cursor = await db.execute(
                    "SELECT file_path, caller, callee, line, line_content FROM function_calls WHERE repo = ? AND (caller = ? OR callee = ?)",
                    (repo, function_name, function_name)
                )
                rows = await cursor.fetchall()
                calls = [
                    {
                        "file_path": row[0],
                        "caller": row[1],
                        "callee": row[2],
                        "line": row[3],
                        "line_content": row[4]
                    }
                    for row in rows
                ]
                await cursor.close()
                logger.debug(f"Retrieved {len(calls)} function calls for {function_name} in repo {repo}")
                return calls
            except Exception as e:
                logger.error(f"Failed to retrieve function calls for {function_name} in repo {repo}: {str(e)}")
                raise

    async def cleanup_session(self, activity_id: str) -> None:
        """Delete all messages for a given activity_id."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute("DELETE FROM messages WHERE activity_id = ?", (activity_id,))
                await db.commit()
                logger.debug(f"Cleaned up messages for activity_id {activity_id}")
            except Exception as e:
                logger.error(f"Failed to clean up messages for activity_id {activity_id}: {str(e)}")
                raise

    async def cleanup_index(self, repo: str) -> None:
        """Delete index metadata for a repo and its associated files."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                cursor = await db.execute("SELECT index_path FROM indexes WHERE repo = ?", (repo,))
                row = await cursor.fetchone()
                if row:
                    index_path = row[0]
                    import os
                    if os.path.exists(index_path):
                        os.remove(index_path)
                    chunk_path = index_path.replace(".faiss", ".json")
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    await db.execute("DELETE FROM indexes WHERE repo = ?", (repo,))
                    await db.execute("DELETE FROM function_calls WHERE repo = ?", (repo,))
                    await db.commit()
                    logger.debug(f"Cleaned up index and function calls for repo {repo}")
                await cursor.close()
            except Exception as e:
                logger.error(f"Failed to clean up index for repo {repo}: {str(e)}")
                raise

    async def retrieve_messages(self, activity_id: str) -> List[Dict[str, Any]]:
            """Retrieve all messages for a given activity_id, ordered by timestamp."""
            await self._init_db()
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    cursor = await db.execute(
                        "SELECT prompt, repo_data, timestamp FROM messages WHERE activity_id = ? ORDER BY timestamp ASC",
                        (activity_id,)
                    )
                    rows = await cursor.fetchall()
                    messages = []
                    for row in rows:
                        repo_data = json.loads(row[1]) if row[1] else None
                        messages.append({
                            "prompt": row[0],
                            "repo_data": repo_data,
                            "timestamp": row[2]
                        })
                    await cursor.close()
                    logger.debug(f"Retrieved {len(messages)} messages for activity_id {activity_id}")
                    return messages
                except Exception as e:
                    logger.error(f"Failed to retrieve messages for activity_id {activity_id}: {str(e)}")
                    raise