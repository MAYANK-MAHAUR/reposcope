import json
from typing import List, Dict, Any

def normalize_repo_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and standardizes a repo chunk for LLM consumption.
    Ensures file path and content are included, truncates long content.
    """
    chunk = {
        "file_path": chunk.get("file_path", "unknown"),
        "content": chunk.get("content", "")
    }
    
    MAX_LENGTH = 500
    if len(chunk["content"]) > MAX_LENGTH:
        chunk["content"] = f"[...content_truncated... {len(chunk['content'])} chars. First {MAX_LENGTH} chars: {chunk['content'][:MAX_LENGTH]}]"

    return chunk

def format_all_chunks_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Applies normalization to all repo chunks and formats them into a clean string for LLM input.
    """
    normalized_chunks = [normalize_repo_chunk(chunk) for chunk in chunks]
    formatted = "\n\n".join([f"File: {chunk['file_path']}\nContent:\n{chunk['content']}" for chunk in normalized_chunks])
    return f"--- REPO CHUNKS START ---\n{formatted}\n--- REPO CHUNKS END ---"

def format_interactions_for_llm(interactions: List[Dict[str, Any]]) -> str:
    """
    Formats conversation history into a string: [USER]: prompt \n\n [REPOSCOPE]: response.
    """
    history_parts = []
    
    for interaction in interactions:
        user_prompt = interaction.get("prompt")
        repo_data = interaction.get("repo_data")
        response = repo_data.get("response") if repo_data else None

        if user_prompt:
            history_parts.append(f"[USER]: {user_prompt.strip()}")
        if response:
            history_parts.append(f"[REPOSCOPE]: {response.strip()}")

    return '\n\n'.join(history_parts)