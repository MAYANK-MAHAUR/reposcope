import asyncio
import aiohttp
import ulid
import json
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("test.log"),
])
logger = logging.getLogger(__name__)
request_id = "01K6BEPKY12FMR1S19Y3SE01C6"
query_id = "01K6BEMZ2QZQ58ADNDCKBPKD51"
activity_id = "01K6BEMNWZFMP3RMGJTFZBND2N"

async def send_request(session: aiohttp.ClientSession, url: str, activity_id: str, prompt: str) -> None:
    """
    Sends a single request to the RepoScopeAgent server and processes the SSE response.
    """
    

    payload = {
        "session": {
            "processor_id": "reposcope-test-client",
            "activity_id": activity_id,
            "request_id": request_id,
            "interactions": []
        },
        "query": {
            "id": query_id,
            "prompt": prompt,
            "context": ""
        }
    }

    print(f"\n=== Sending Prompt: {prompt} ===")
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                print(f"❌ Server responded with status {resp.status}")
                text = await resp.text()
                print("Server response content:", text)
                logger.error(f"Server error: status={resp.status}, content={text}")
                return

            current_event = None
            last_chunk_event = None
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    if not current_event.endswith("_CHUNK") or current_event != last_chunk_event:
                        print(f"\n--- 📝 Block Received: {current_event} ---")
                        logger.debug(f"Received event: {current_event}")
                        last_chunk_event = current_event if current_event.endswith("_CHUNK") else None
                elif line.startswith("data:") and current_event:
                    try:
                        data = json.loads(line[5:].strip())
                        if 'content' in data:
                            print(data['content'], end='', flush=True)
                            logger.debug(f"Received data content: {data['content']}")
                        elif data.get('event') == 'complete':
                            print(f"\n\n--- ✅ {current_event} COMPLETE ---")
                            logger.debug(f"Event {current_event} completed")
                            current_event = None
                            last_chunk_event = None
                        elif data.get('event') == 'error':
                            print(f"\n❌ Error: {data.get('error_message', 'Unknown error')}")
                            logger.error(f"Server error event: {data}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse SSE data: {line}, error: {str(e)}")
                        print(f"\n❌ Invalid SSE data: {line}")
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Could not connect to the server: {e}")
        print("Please ensure your server.py is running on http://127.0.0.1:8000.")
        logger.error(f"Connection error: {str(e)}")

async def run_test_suite():
    """
    Runs a predefined suite of test prompts to verify RepoScopeAgent functionality.
    """
    url = "http://127.0.0.1:8000/assist"
    test_prompts = [
        "set default github.com/pallets/flask",
    ]

    print(f"\n=== Starting RepoScopeAgent Test Suite (Session ID: {activity_id}) ===")
    async with aiohttp.ClientSession() as session:
        for prompt in test_prompts:
            await send_request(session, url, activity_id, prompt)
            await asyncio.sleep(2)  # Avoid overwhelming the server
        print("\n=== Test Suite Completed ===")

async def interactive_chat():
    """
    Runs an interactive chat session, allowing continuous user input until 'exit' or 'quit'.
    """
    url = "http://127.0.0.1:8000/assist"
    print(f"\n=== Starting RepoScopeAgent Chat (Session ID: {activity_id}) ===")
    print("Type your prompt (e.g., 'set default github.com/pallets/flask' or 'exit' to quit):")

    async with aiohttp.ClientSession() as session:
        while True:
            prompt = input("> ").strip()
            if prompt.lower() in ["exit", "quit"]:
                await send_request(session, url, activity_id, "exit chat")
                print("Chat session ended. Session data cleared.")
                break
            if not prompt:
                print("Please enter a valid prompt or type 'exit' to quit.")
                continue
            await send_request(session, url, activity_id, prompt)
            await asyncio.sleep(1)  # Avoid overwhelming the server

if __name__ == "__main__":
    asyncio.run(run_test_suite())
    asyncio.run(interactive_chat())