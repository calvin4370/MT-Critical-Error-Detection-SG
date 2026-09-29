# The project's Python environment from the locked dependencies (uv.lock), so it is the
# same on any machine. GROUPS picks optional dependency groups:
#   docker build -t linguasg-app --build-arg GROUPS="--group app" .     (web app)
#   docker build -t linguasg-train --build-arg GROUPS="--group train" . (GPU training)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
ARG GROUPS=""

# Dependencies first: this layer is reused when only the code changes
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --locked --no-install-project $GROUPS
COPY src ./src
COPY configs ./configs
RUN uv sync --locked $GROUPS

ENTRYPOINT ["uv", "run", "--no-sync"]
