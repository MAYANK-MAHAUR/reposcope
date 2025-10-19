import git
import os
import aiohttp
import logging
import shutil
import tempfile
from typing import List, Dict, Any
from config import GITHUB_API_BASE
from tree_sitter import Language, Parser, Query
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO
import re

logger = logging.getLogger(__name__)

try:
    PY_LANGUAGE = Language(tspython.language())
    JS_LANGUAGE = Language(tsjavascript.language())
    py_parser = Parser(PY_LANGUAGE)
    js_parser = Parser(JS_LANGUAGE)
    logger.debug("tree-sitter initialized for Python and JavaScript")
except Exception as e:
    logger.error(f"tree-sitter initialization failed: {str(e)}. Falling back to regex parsing.")
    py_parser = None
    js_parser = None

async def clone_repo(repo_url: str, repo_path: str) -> None:
    """Clone a GitHub repo to a local path (full clone)."""
    try:
        repo_path = os.path.join(tempfile.gettempdir(), os.path.basename(repo_path))
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
        git.Repo.clone_from(repo_url, repo_path)
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
                    logger.debug(f"Fetched file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {str(e)}")
    return files

async def count_lines_of_code(repo_path: str) -> Dict[str, Any]:
    """Count lines of code in a repo, categorized by file type."""
    total_lines = 0
    python_lines = 0
    js_lines = 0
    other_lines = 0
    python_files = 0
    js_files = 0
    other_files = 0

    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith(('.py', '.js', '.java', '.md', '.txt')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.read().splitlines())
                    total_lines += lines
                    if filename.endswith('.py'):
                        python_lines += lines
                        python_files += 1
                    elif filename.endswith('.js'):
                        js_lines += lines
                        js_files += 1
                    else:
                        other_lines += lines
                        other_files += 1
                except Exception as e:
                    logger.error(f"Failed to count lines in {file_path}: {str(e)}")
    return {
        "total_lines": total_lines,
        "python_lines": python_lines,
        "python_files": python_files,
        "js_lines": js_lines,
        "js_files": js_files,
        "other_lines": other_lines,
        "other_files": other_files
    }

async def parse_function_calls(repo_path: str, repo_name: str = None) -> List[Dict[str, Any]]:
    """Parse Python and JavaScript files for function definitions and calls."""
    if not repo_name:
        logger.warning("No repo_name provided to parse_function_calls. Using default 'unknown_repo'.")
        repo_name = "unknown_repo"
    function_calls = []
    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith(('.py', '.js')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    parser = py_parser if filename.endswith('.py') else js_parser
                    language = PY_LANGUAGE if filename.endswith('.py') else JS_LANGUAGE
                    if parser:
                        tree = parser.parse(content.encode('utf-8'))
                        function_calls.extend(extract_functions_and_calls(tree, file_path, content, repo_name, language))
                    else:
                        logger.warning(f"Using regex fallback for {file_path} due to tree-sitter failure.")
                        function_calls.extend(extract_functions_and_calls_regex(file_path, content, repo_name))
                except Exception as e:
                    logger.error(f"Failed to parse file {file_path}: {str(e)}")
    return function_calls

def extract_functions_and_calls(tree, file_path: str, content: str, repo_name: str, language: Language) -> List[Dict[str, Any]]:
    """Extract function definitions and calls from a tree-sitter parse tree."""
    calls = []
    lines = content.splitlines()

    def get_text(node):
        start = node.start_byte
        end = node.end_byte
        return content[start:end]

    def traverse(node, current_function=None):
        node_type = 'function_definition' if language == PY_LANGUAGE else 'function_declaration'
        call_type = 'call' if language == PY_LANGUAGE else 'call_expression'
        if node.type == node_type:
            func_name = node.child_by_field_name('name')
            if func_name:
                current_function = get_text(func_name)
        elif node.type == call_type:
            func_node = node.child_by_field_name('function')
            if func_node:
                called_func_full = get_text(func_node)
                called_func = called_func_full.split('.')[-1]
                start_line = node.start_point[0]
                line_content = lines[start_line] if start_line < len(lines) else ""
                calls.append({
                    "file_path": file_path,
                    "caller": current_function,
                    "callee": called_func,
                    "line": start_line + 1,
                    "line_content": line_content.strip(),
                    "repo": repo_name
                })
        for child in node.children:
            traverse(child, current_function)

    traverse(tree.root_node)
    return calls

def extract_functions_and_calls_regex(file_path: str, content: str, repo_name: str) -> List[Dict[str, Any]]:
    """Fallback: Extract function calls using regex (less accurate)."""
    calls = []
    lines = content.splitlines()
    current_function = None
    func_def_pattern = re.compile(r'^\s*(def|function)\s+([a-zA-Z_]\w*)\s*\(', re.MULTILINE)
    func_call_pattern = re.compile(r'([a-zA-Z_]\w*)\s*\(', re.MULTILINE)

    for i, line in enumerate(lines):
        def_match = func_def_pattern.match(line)
        if def_match:
            current_function = def_match.group(2)
            continue
        call_matches = func_call_pattern.finditer(line)
        for match in call_matches:
            called_func = match.group(1)
            calls.append({
                "file_path": file_path,
                "caller": current_function,
                "callee": called_func,
                "line": i + 1,
                "line_content": line.strip(),
                "repo": repo_name
            })
    return calls

async def find_code_patterns(repo_path: str, pattern: str, repo_name: str = None) -> List[Dict[str, Any]]:
    """Search for code patterns (e.g., try-except, try-catch) in Python and JavaScript files."""
    if not repo_name:
        repo_name = "unknown_repo"
    matches = []
    py_query = """
    (try_statement
        block: (block) @try_block
        (except_clause)* @except_clause)
    (try_statement
        block: (block) @try_block
        except_clause: (except_clause
            (identifier)? @exception))
    """
    js_query = """
    (try_statement
        block: (statement_block) @try_block
        (catch_clause)* @catch_clause)
    (try_statement
        block: (statement_block) @try_block
        catch_clause: (catch_clause
            (identifier)? @exception))
    """

    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith(('.py', '.js')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    parser = py_parser if filename.endswith('.py') else js_parser
                    language = PY_LANGUAGE if filename.endswith('.py') else JS_LANGUAGE
                    query = py_query if filename.endswith('.py') else js_query
                    if parser and pattern in ('try-except', 'try-catch'):
                        tree = parser.parse(content.encode('utf-8'))
                        query_obj = Query(language, query)
                        captures = query_obj.captures(tree.root_node)
                        lines = content.splitlines()
                        for capture in captures:
                            node, name = capture
                            start_line = node.start_point[0]
                            line_content = lines[start_line] if start_line < len(lines) else ""
                            matches.append({
                                "file_path": file_path,
                                "pattern": pattern,
                                "line": start_line + 1,
                                "line_content": line_content.strip(),
                                "repo": repo_name
                            })
                        logger.debug(f"Searched {file_path} for {pattern}: found {len(captures)} matches")
                        if captures:
                            logger.info(f"Found {pattern} in {file_path}: {[lines[node.start_point[0]].strip() for node, _ in captures]}")
                    else:
                        pattern_re = re.compile(r'\btry\b.*?\bexcept\b' if filename.endswith('.py') else r'\btry\b.*?\bcatch\b', re.DOTALL)
                        for match in pattern_re.finditer(content):
                            start_line = content[:match.start()].count('\n') + 1
                            line_content = content[match.start():match.end()].split('\n')[0]
                            matches.append({
                                "file_path": file_path,
                                "pattern": pattern,
                                "line": start_line,
                                "line_content": line_content.strip(),
                                "repo": repo_name
                            })
                        logger.debug(f"Searched {file_path} for {pattern} with regex: found {len(pattern_re.findall(content))} matches")
                        if pattern_re.findall(content):
                            logger.info(f"Found {pattern} (regex) in {file_path}: {pattern_re.findall(content)[0][:50]}...")
                except Exception as e:
                    logger.error(f"Failed to search patterns in {file_path}: {str(e)}")
    return matches

async def run_pylint(repo_path: str, repo_name: str = None) -> List[Dict[str, Any]]:
    """Run pylint on Python files in the repo and collect issues."""
    if not repo_name:
        repo_name = "unknown_repo"
    issues = []
    output = StringIO()
    reporter = TextReporter(output=output)

    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                try:
                    Run([file_path, '--disable=missing-module-docstring,missing-class-docstring,missing-function-docstring'], reporter=reporter, do_exit=False)
                    output.seek(0)
                    lines = output.readlines()
                    for line in lines:
                        match = re.match(r'(.+):(\d+):(\d+): (\w+): (.+)', line.strip())
                        if match:
                            file_path, line_num, _, issue_code, message = match.groups()
                            issues.append({
                                "file_path": file_path,
                                "line": int(line_num),
                                "issue_code": issue_code,
                                "message": message,
                                "repo": repo_name
                            })
                    output.truncate(0)
                    output.seek(0)
                except Exception as e:
                    logger.error(f"Failed to run pylint on {file_path}: {str(e)}")
    return issues