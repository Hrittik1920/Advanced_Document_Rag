FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN useradd -m sandboxuser
WORKDIR /app

# Add fastapi and an ASGI server like uvicorn
RUN uv pip install --system numpy pandas fastapi uvicorn pydantic

COPY --chown=sandboxuser:sandboxuser runner.py .
USER sandboxuser

# Run the API server on port 8000
CMD ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "9999"]