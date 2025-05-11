# Use the official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Default port
ENV PORT=8001

# Expose the port
EXPOSE $PORT

# Create an entrypoint script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'python3 -m chainlit run app.py --host 0.0.0.0 --port $PORT' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Use the entrypoint script
CMD ["/app/entrypoint.sh"]
