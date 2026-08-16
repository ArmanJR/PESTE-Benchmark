FROM ghcr.io/astral-sh/uv:0.8.14 AS uv
FROM nvcr.io/nvidia/pytorch@sha256:3cb18e2c438db8af2d3a659ca27fac5da328640261c38c48a34edcd223c38af9

ARG PESTE_SOURCE_REVISION=unknown
ARG TORCHAUDIO_REVISION=6e1c7fe9ff6d82b8665d0a46d859d3357d2ebaaa
LABEL org.opencontainers.image.source="https://github.com/ArmanJR/PESTE-Benchmark" \
      org.opencontainers.image.revision="$PESTE_SOURCE_REVISION" \
      org.opencontainers.image.version="2.1.0"
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PESTE_PROJECT_ROOT=/opt/peste \
    PESTE_SOURCE_REVISION=$PESTE_SOURCE_REVISION \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venvs/modern/bin:$PATH

COPY runtimes/modern/pyproject.toml runtimes/modern/uv.lock /opt/runtime/modern/
COPY runtimes/nemo/pyproject.toml runtimes/nemo/uv.lock /opt/runtime/nemo/
RUN uv venv --python /usr/bin/python3 --system-site-packages /opt/venvs/modern \
    && uv export --directory /opt/runtime/modern --quiet --frozen --no-dev --no-emit-project \
      --prune torch --prune torchaudio --prune torchvision --prune torchcodec \
      --output-file /tmp/modern-requirements.txt \
    && uv pip install --python /opt/venvs/modern/bin/python --no-deps \
      --requirement /tmp/modern-requirements.txt \
    && uv venv --python /usr/bin/python3 /opt/venvs/nemo \
    && test -d /usr/local/lib/python3.12/dist-packages/torch \
    && ln -s /usr/local/lib/python3.12/dist-packages/torch \
      /opt/venvs/nemo/lib/python3.12/site-packages/torch \
    && for peste_path in /usr/local/lib/python3.12/dist-packages/torch-*.dist-info; do \
      test -d "$peste_path"; \
      ln -s "$peste_path" /opt/venvs/nemo/lib/python3.12/site-packages/; \
    done \
    && for peste_package in functorch torchgen; do \
      if test -d "/usr/local/lib/python3.12/dist-packages/$peste_package"; then \
        ln -s "/usr/local/lib/python3.12/dist-packages/$peste_package" \
          "/opt/venvs/nemo/lib/python3.12/site-packages/$peste_package"; \
      fi; \
    done \
    && uv export --directory /opt/runtime/nemo --quiet --frozen --no-dev --no-emit-project \
      --prune torch --prune torchaudio --prune torchvision --prune torchcodec \
      --output-file /tmp/nemo-requirements.txt \
    && uv pip install --python /opt/venvs/nemo/bin/python --no-deps \
      --requirement /tmp/nemo-requirements.txt \
    && USE_CUDA=1 uv pip install --python /opt/venvs/nemo/bin/python --no-deps \
      --no-build-isolation \
      "torchaudio @ git+https://github.com/pytorch/audio.git@$TORCHAUDIO_REVISION"

WORKDIR /opt/peste
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY models ./models
COPY suites ./suites
COPY runtimes ./runtimes
RUN uv pip install --python /opt/venvs/modern/bin/python --no-deps . \
    && uv pip install --python /opt/venvs/nemo/bin/python --no-deps . \
    && /opt/venvs/modern/bin/python -c 'import numpy,torch; from peft.import_utils import is_torchao_available; from transformers import AutoModelForCTC,AutoModelForMultimodalLM,AutoProcessor,GenerationMixin; assert torch.__version__.startswith("2.8.0a0+5228986c39"); assert torch.distributed.is_available(); assert is_torchao_available(); assert all((AutoModelForCTC,AutoModelForMultimodalLM,AutoProcessor,GenerationMixin)); torch.from_numpy(numpy.zeros(1))' \
    && /opt/venvs/nemo/bin/python -c 'import numpy,torch,torchaudio; from nemo.collections.asr.models import ASRModel; assert torch.__version__.startswith("2.8.0a0+5228986c39"); assert torch.distributed.is_available(); assert torchaudio.__version__.startswith("2.8.0"); assert ASRModel; torch.from_numpy(numpy.zeros(1))'

COPY runtimes/offline_guard.c /tmp/offline_guard.c
RUN install -d -m 0755 /opt/peste/lib /cache/dataset /cache/hf /results /home/peste \
    && cc -shared -fPIC -O2 -Wall -Wextra -Werror \
      -o /opt/peste/lib/libpeste_offline.so /tmp/offline_guard.c -ldl \
    && groupadd --gid 10001 peste \
    && useradd --uid 10001 --gid 10001 --home-dir /home/peste \
      --shell /usr/sbin/nologin peste \
    && chown peste:peste /home/peste \
    && chmod 0755 /cache /cache/dataset /cache/hf /results

ENTRYPOINT []
CMD ["peste", "--help"]
