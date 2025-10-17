import git
import os
import aiohttp
import logging
import shutil
import tempfile
from typing import List, Dict, Any
from config import GITHUB_API_BASE
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)

# Initialize tree-sitter for Python
# Uses tspython.language() to get the correct handle for the Language constructor (Fix for TypeError)
PY_LANGUAGE = Language(tspython.language())
# Passes the language directly to the Parser constructor (Fix for AttributeError: 'Parser' object has no attribute 'set_language')
parser = Parser(PY_LANGUAGE)

async def clone_repo(repo_url: str, repo_path: str) -> None:
    """Clone a GitHub repo to a local path (shallow clone for speed)."""
    try:
        repo_path = os.path.join(tempfile.gettempdir(), os.path.basename(repo_path))
        if os.path.exists(repo_path):
            # NOTE: Removed ignore_errors=True in a previous step to better catch file lock issues
            shutil.rmtree(repo_path, ignore_errors=True) 
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        logger.debug(f"Cloned repo to {repo_path}")
    except Exception as e:
        logger.error(f"Failed to clone repo {repo_url}: {str(e)}")
        raise

async def fetch_repo_files(repo_path: str) -> List[Dict[str, str]]:
    """Read files from a cloned repo, returning file paths and contents."""
    files = []
    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith(('.py', '.js', '.java', '.md', '.txt')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    files.append({"file_path": file_path, "content": content})
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {str(e)}")
    return files

async def parse_function_calls(repo_path: str) -> List[Dict[str, Any]]:
    """Parse Python files for function definitions and calls."""
    function_calls = []
    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    tree = parser.parse(content.encode('utf-8'))
                    function_calls.extend(extract_functions_and_calls(tree, file_path, content))
                except Exception as e:
                    logger.error(f"Failed to parse file {file_path}: {str(e)}")
    return function_calls

def extract_functions_and_calls(tree, file_path: str, content: str) -> List[Dict[str, Any]]:
    """Extract function definitions and calls from a tree-sitter parse tree."""
    calls = []
    lines = content.splitlines()

    def get_text(node):
        start = node.start_byte
        end = node.end_byte
        return content[start:end]

    def traverse(node, current_function=None):
        if node.type == 'function_definition':
            func_name = node.child_by_field_name('name')
            if func_name:
                current_function = get_text(func_name)
        elif node.type == 'call':
            func_node = node.child_by_field_name('function')
            if func_node:
                # 1. Get the full name (e.g., 'app.route' or just 'route')
                called_func_full = get_text(func_node)
                
                # 2. Normalize the name by taking only the last part (Fix for "No calls found for route")
                called_func = called_func_full.split('.')[-1]
                
                start_line = node.start_point[0]
                line_content = lines[start_line] if start_line < len(lines) else ""
                calls.append({
                    "file_path": file_path,
                    "caller": current_function,
                    "callee": called_func, # Stores the normalized function name
                    "line": start_line + 1,
                    "line_content": line_content.strip()
                })
        for child in node.children:
            traverse(child, current_function)

    traverse(tree.root_node)
    return calls

async def fetch_repo_via_api(owner: str, repo: str) -> List[Dict[str, str]]:
    """Fetch repo file contents via GitHub API (alternative to cloning)."""
    async with aiohttp.ClientSession() as session:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ValueError(f"GitHub API error: {resp.status}")
                contents = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch repo {owner}/{repo}: {str(e)}")
            raise

        files = []
        for item in contents:
            if item["type"] == "file" and item["name"].endswith(('.py', '.js', '.java', '.md', '.txt')):
                async with session.get(item["download_url"]) as file_resp:
                    if file_resp.status == 200:
                        content = await file_resp.text()
                        files.append({"file_path": item["path"], "content": content})
        return files