FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy all application source code
COPY requirements.txt .
COPY agent-concurrent/ ./agent-concurrent/
COPY agent-core/ ./agent-core/
COPY agent-engine/ ./agent-engine/
COPY agent-persist/ ./agent-persist/
COPY agent-sim/ ./agent-sim/

# 2. Copy the entrypoint script to a standard binary location, NOT /app.
COPY agent-sim/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# 3. Give it execute permissions in its new location.
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 4. Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p data logs

# 5. Set the ENTRYPOINT to the new, safe location.
ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["echo", "This default command is overridden by docker-compose.yml"]
