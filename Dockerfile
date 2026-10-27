FROM python:3.12-slim

# ── Build arguments ───────────────────────────────────────────────────────────
ARG HAPHAZARD_VERSION=""
ARG EXTRAS=all
ARG DEFAULT_SEED=42
# Keep these in sync with the pins in setup.py (torch / torchvision)
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
# CPU by default. For GPU, pass a CUDA index that exists for TORCH_VERSION, e.g.
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

LABEL org.opencontainers.image.title="haphazard" \
      org.opencontainers.image.source="https://github.com/theArijitDas/Haphazard-Package/" \
      org.opencontainers.image.licenses="MIT"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONHASHSEED=${DEFAULT_SEED} \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --create-home --uid 1000 haphazard
WORKDIR /workspace
RUN chown haphazard:haphazard /workspace

# ── PyTorch (pinned to match setup.py so pip won't replace it later) ──────────
RUN pip install --upgrade pip && \
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url ${TORCH_INDEX}

# ── haphazard (pinned version + chosen extras) ────────────────────────────────
# If HAPHAZARD_VERSION is empty, the latest release is installed.
# A C compiler is needed only at build time: tdigest -> accumulation-tree has no
# prebuilt wheel and compiles a Cython extension. Installed and removed in the
# same layer so it doesn't add to the final image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install "haphazard[${EXTRAS}]${HAPHAZARD_VERSION:+==${HAPHAZARD_VERSION}}" && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

USER haphazard

CMD ["/bin/bash", "-c", "python -c 'import haphazard; print(\"haphazard ready\")' && exec /bin/bash"]