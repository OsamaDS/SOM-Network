# Base Python image
FROM python:3.11-slim

WORKDIR /app

# Install pip-tools for dependency management
RUN pip install --no-cache-dir pip-tools

# Copy project files
COPY . .

# Install uv package
RUN pip install --no-cache-dir uv

# Install dependencies via uv
RUN uv pip install --system .

# Expose API port
EXPOSE 9000

# Default command to run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
