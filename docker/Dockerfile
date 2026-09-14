# syntax=docker/dockerfile:1

# --- Builder: install the project and its dependencies into a virtualenv ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# --- Runtime: copy only the built virtualenv, run as a non-root user ---
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 microtwin
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pre-create /data owned by the non-root user: a named volume mounted here
# inherits this directory's ownership on first initialization, otherwise
# Docker mounts it root-owned and the app can't write to it.
RUN mkdir -p /data && chown microtwin:microtwin /data

USER microtwin

EXPOSE 8000

CMD ["uvicorn", "microtwin.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
