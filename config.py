import os
from dotenv import load_dotenv
import openai

load_dotenv()

# GaiaNet LLM configuration
GAIA_API_KEY = os.getenv("GAIA_API_KEY", "gaia-ZWFlMGYwNmQtNGVmYS00YmU5LTg1NGUtNzFlOTM3NWU3YzU2-cFu80IEd7q0m2z7j")
GAIA_API_BASE = os.getenv("GAIA_API_BASE", "https://qwen72b.gaia.domains/v1")

async_client = openai.AsyncOpenAI(
    base_url=GAIA_API_BASE,
    api_key=GAIA_API_KEY
)

# GitHub API base (no auth needed for public repos)
GITHUB_API_BASE = "https://api.github.com"