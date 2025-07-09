import asyncio
import os

from agent_sim.infrastructure.database.models import create_tables
from dotenv import load_dotenv

# Load environment variables from .env file
# This is important so the script can find the DATABASE_URL
project_root = os.path.join(os.path.dirname(__file__), "../../../../..")
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)


async def main():
    """The main entry point for the database initialization script."""
    print("--- Database Initializer ---")
    print("Attempting to connect to the database and create tables...")
    try:
        await create_tables()
        print("Success: Database tables created or already exist.")
    except Exception as e:
        print(f"Error: Could not initialize database. Reason: {e}")
        # Exit with a non-zero status code to indicate failure
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
