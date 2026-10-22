# CPU image: laptops, HPC CPU nodes (via Apptainer), AWS/GCP Batch, Kubernetes.
#   docker build -f deploy/Dockerfile -t featurescope:cpu .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends rsync openssh-client curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip \
    && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/aws.zip
WORKDIR /opt/featurescope
COPY pyproject.toml README.md ./
COPY featurescope ./featurescope
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[batch]"
ENV NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl
EXPOSE 8765
CMD ["featurescope", "serve", "--host", "0.0.0.0", "--workspace", "/data"]
