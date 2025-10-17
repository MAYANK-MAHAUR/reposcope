from sentient_agent_framework import DefaultServer
from agent import RepoScopeAgent

if __name__ == "__main__":
    agent = RepoScopeAgent()
    server = DefaultServer(agent=agent)
    print("🚀 RepoScope server running at http://127.0.0.1:8000")
    server.run()