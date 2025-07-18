FROM python:3.11-slim

# Install build dependencies including git for osqp
RUN apt-get update && apt-get install -y gcc g++ git graphviz graphviz-dev netcat-traditional && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy and install external dependencies during build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the entrypoint script (not the source code)
COPY agent-sim/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories
RUN mkdir -p data logs

# Source code will be available via volume mount
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["echo", "This default command is overridden by docker-compose.yml"]