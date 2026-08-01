FROM ghcr.io/astral-sh/uv:0.8.14 AS uv
FROM nvcr.io/nvidia/pytorch@sha256:90f3c17838fde28d5c7ae2d5bfbc8a4c587d3797767ea96cdd48fe82e3613f3b

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PESTE_PROJECT_ROOT=/opt/peste \
    PYTHONPATH=/opt/peste/runtimes/transformers-compat \
    PYTHONUNBUFFERED=1
WORKDIR /opt/runtime
COPY runtimes/vibevoice/pyproject.toml runtimes/vibevoice/uv.lock ./
RUN uv venv --python /usr/bin/python3 --system-site-packages /opt/venv \
    && uv export --quiet --frozen --no-dev --no-emit-project \
      --prune torch --prune torchaudio --prune torchvision --prune torchcodec \
      --output-file /tmp/runtime-requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps \
      --requirement /tmp/runtime-requirements.txt \
    && /opt/venv/bin/python -c 'import numpy,torch; assert torch.__version__.startswith("2.8.0a0+5228986"); torch.from_numpy(numpy.zeros(1))'
WORKDIR /opt/peste
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY models ./models
COPY suites ./suites
COPY runtimes ./runtimes
RUN uv pip install --python /opt/venv/bin/python --no-deps . \
    && /opt/venv/bin/python -c 'from transformers import AutoProcessor,GenerationMixin; assert all((AutoProcessor,GenerationMixin))'

ENTRYPOINT []
CMD ["peste", "--help"]
