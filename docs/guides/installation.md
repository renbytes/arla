# Installation & Setup

Welcome to the ARLA framework! This guide will walk you through the process of setting up the project on your local machine. The primary method for running ARLA is through Docker, which ensures a consistent and reproducible environment.

## Prerequisites

Before you begin, please ensure you have the following software installed on your system:

- **Git**: For cloning the repository.
- **Docker & Docker Compose**: For running the simulation environment.
- **An OpenAI API Key**: The cognitive systems in ARLA rely on Large Language Models.

## Step 1: Clone the Repository

First, clone the official ARLA repository from GitHub to your local machine.

```bash
git clone https://github.com/renbytes/arla.git
cd arla
```

## Step 2: Configure Your Environment

The project uses a `.env` file to manage sensitive information like API keys and database credentials. We've provided an example file to get you started.

1. **Copy the example file:**

```bash
cp .env.example .env
```

2. **Edit the `.env` file**: Open the newly created `.env` file in your favorite text editor. You will need to add your OpenAI API key:

```bash
# ... (other settings)

# --- OpenAI API Key (Developer must provide their own) ---
OPENAI_API_KEY=sk-YourSecretKeyGoesHere
```

## Step 3: Build and Start the Services

The entire ARLA environment, including the application, database, and MLflow server, is managed by Docker Compose. The provided `Makefile` simplifies this process.

Run the following command from the root of the project:

```bash
make up
```

This command will:

- Build the Docker images for the application and worker services.
- Download the official images for PostgreSQL, Redis, and MLflow.
- Start all the services in the background.

The first time you run this, it may take several minutes to download the images and install all the dependencies. Subsequent runs will be much faster thanks to Docker's caching.

## Step 4: Verify the Installation

Once the services are running, you can verify that everything is working correctly.

1. **Check the running containers:**

```bash
docker compose ps
```

You should see all services (app, worker, db, redis, mlflow) listed with a `healthy` status.

2. **Run an example simulation:** The repository includes a command to run a small, pre-configured simulation.

```bash
make run-example
```

If the installation was successful, you will see the simulation start and log output to your terminal.

Congratulations! You now have a complete ARLA development environment running locally. You're ready to start running experiments or developing your own simulations.
