# Installation & Setup

Get ARLA running on your local machine in minutes. Our Docker-based setup ensures consistency across all platforms while providing a complete development environment with cognitive systems, data persistence, and experiment tracking.

!!! info "What You'll Build"
    By the end of this guide, you'll have a complete ARLA development environment with:
    
    - **Agent simulation engine** with cognitive systems
    - **PostgreSQL database** for experiment data and agent memories
    - **MLflow tracking server** for experiment management and visualization
    - **Redis message broker** for distributed computing and task queues

## Prerequisites

Before you begin, ensure you have these tools installed on your development machine:

<div class="grid cards" markdown>

-   :fontawesome-brands-git-alt:{ .lg .middle } **Git**

    ---

    For cloning the repository and version control.

    [Download Git](https://git-scm.com/downloads){ .md-button }

-   :fontawesome-brands-docker:{ .lg .middle } **Docker & Docker Compose**

    ---

    For running the containerized development environment.

    [Download Docker](https://docs.docker.com/get-docker/){ .md-button }

-   :material-key:{ .lg .middle } **OpenAI API Key**

    ---

    Required for cognitive systems that use Large Language Models.

    [Get API Key](https://platform.openai.com/api-keys){ .md-button }

</div>

## Quick Start

### Step 1: Clone the Repository

Get the latest version of ARLA from GitHub:

=== "HTTPS"

    ```bash
    git clone https://github.com/renbytes/arla.git
    cd arla
    ```

=== "SSH"

    ```bash
    git clone git@github.com:renbytes/arla.git
    cd arla
    ```

=== "GitHub CLI"

    ```bash
    gh repo clone renbytes/arla
    cd arla
    ```

### Step 2: Configure Environment

ARLA uses environment variables for sensitive configuration like API keys and database credentials.

**Create Configuration File:**

The `Makefile` includes a convenient setup command:

```bash
make setup
```

This copies `.env.example` to `.env` with sensible defaults.

**Add Your API Key:**

Open the newly created `.env` file and add your OpenAI API key:

```bash title=".env"
# --- OpenAI API Key (Required) ---
OPENAI_API_KEY=sk-YourSecretKeyGoesHere

# --- Database Configuration ---
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_DB=agent_sim_db

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=password

# --- Redis Configuration ---
REDIS_URL=redis://redis:6379/0
```

!!! warning "Security Note"
    Never commit your `.env` file to version control. It's already included in `.gitignore`.

### Step 3: Build and Start Services

Launch the complete ARLA environment with a single command:

```bash
make up
```

**What Happens During Setup:**

<div class="grid cards" markdown>

-   **1. Image Building**

    ---

    Docker builds custom images for the application and worker services, installing Python dependencies with Poetry.

-   **2. Service Startup**

    ---

    Launches PostgreSQL database, Redis message broker, and MLflow tracking server with health checks.

-   **3. Dependency Caching**

    ---

    Creates shared volumes to cache Python dependencies, making subsequent builds much faster.

-   **4. Database Migration**

    ---

    Automatically creates database tables and applies any pending migrations.

</div>

!!! tip "First Run Performance"
    The initial setup may take 5-10 minutes to download images and install dependencies. Subsequent runs are much faster thanks to Docker's caching.

### Step 4: Verify Installation

Confirm everything is working correctly:

**Check Service Status:**

```bash
docker compose ps
```

You should see all services running with `healthy` status:

```
NAME            STATUS                    PORTS
arla-app-1      Up (healthy)             
arla-worker-1   Up (healthy)             
arla-db-1       Up (healthy)             0.0.0.0:5432->5432/tcp
arla-redis-1    Up (healthy)             0.0.0.0:6379->6379/tcp
arla-mlflow-1   Up (healthy)             0.0.0.0:5001->5000/tcp
```

**Run Example Simulation:**

Test the installation with a pre-configured simulation:

```bash
make run-example
```

**Expected Output:**
```
✅ Database connection successful
📊 MLflow tracking enabled
🤖 Spawning 10 agents in 50x50 grid world
⚡ Starting simulation with 1000 ticks...
🧠 Cognitive systems initialized: Reflection, Q-Learning, Identity, Goals
🎬 Simulation running... (Tick 1/1000)
```

**Access MLflow UI:**

Open your browser and navigate to [http://localhost:5001](http://localhost:5001) to view the experiment tracking interface.

!!! success "Installation Complete!"
    You now have a complete ARLA development environment running locally. You're ready to start building simulations!

---

## Development Workflow

### Daily Development Commands

Once installed, use these commands for daily development:

```bash
# Start services
make up

# Run a quick test simulation
make run-example

# Run the full test suite
make test

# View logs from all services
make logs
```