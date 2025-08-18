# Installation & Setup

Get ARLA running on your local machine in minutes. Our Docker-based setup ensures consistency across all platforms.

!!! info "What You'll Build"
    By the end of this guide, you'll have a complete ARLA development environment with:
    
    - **Agent simulation engine** with cognitive systems
    - **PostgreSQL database** for experiment data
    - **MLflow tracking server** for experiment management
    - **Redis message broker** for distributed computing

## Prerequisites

Before you begin, ensure you have these tools installed:

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

## Step 1: Clone the Repository

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

## Step 2: Configure Environment

ARLA uses environment variables for sensitive configuration like API keys.

### Create Configuration File

The `Makefile` includes a convenient setup command:

```bash
make setup
```

This copies `.env.example` to `.env` with default values.

### Add Your API Key

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
```

!!! warning "API Key Security"
    Never commit your `.env` file to version control. It's already included in `.gitignore`.

## Step 3: Build and Start Services

Launch the complete ARLA environment with a single command:

```bash
make up
```

### What Happens During Setup

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

</div>

!!! tip "First Run Performance"
    The initial setup may take 5-10 minutes to download images and install dependencies. Subsequent runs are much faster thanks to Docker's caching.

## Step 4: Verify Installation

Confirm everything is working correctly:

### Check Service Status

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

### Run Example Simulation

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
```

### Access MLflow UI

Open your browser and navigate to [http://localhost:5001](http://localhost:5001) to view the experiment tracking interface.

!!! success "Installation Complete!"
    You now have a complete ARLA development environment running locally. You're ready to start building simulations!

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Build Your First Simulation**

    ---

    Follow our step-by-step tutorial to create a custom simulation from scratch.

    [:octicons-arrow-right-24: Start Tutorial](../tutorials/first-simulation.md)

-   :material-play:{ .lg .middle } **Run Experiments**

    ---

    Learn how to design and execute large-scale experiments with multiple configurations.

    [:octicons-arrow-right-24: Running Simulations](running-simulations.md)

-   :material-cogs:{ .lg .middle } **Architecture Deep Dive**

    ---

    Understand ARLA's modular design and cognitive architecture components.

    [:octicons-arrow-right-24: Architecture Guide](../architecture/index.md)

</div>

## Troubleshooting

### Common Issues

??? failure "Docker Services Won't Start"
    
    **Problem**: Services fail to start or remain unhealthy
    
    **Solutions**:
    ```bash
    # Check if ports are already in use
    netstat -tlnp | grep ':5432\|:6379\|:5001'
    
    # Reset and rebuild
    docker compose down -v
    docker compose build --no-cache
    docker compose up -d
    ```

??? failure "MLflow Authentication Errors"
    
    **Problem**: 401 Unauthorized when accessing MLflow UI
    
    **Solutions**:
    - Verify credentials in `.env` file match MLflow configuration
    - Check MLflow logs: `docker compose logs mlflow`
    - Restart MLflow service: `docker compose restart mlflow`

??? failure "API Key Not Working"
    
    **Problem**: OpenAI API errors during simulation
    
    **Solutions**:
    - Verify API key is valid and has credits
    - Check the key starts with `sk-`
    - Ensure no extra spaces in `.env` file

### Getting Help

- **Documentation**: Browse our comprehensive guides
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions in our GitHub repository

---

**Ready to build your first simulation?** Follow our [tutorial guide](../tutorials/first-simulation.md) to create agents from scratch.