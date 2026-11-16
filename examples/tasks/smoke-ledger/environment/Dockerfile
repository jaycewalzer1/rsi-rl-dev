FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
WORKDIR /app
COPY app/ /app/
RUN git init -q && git -c user.email=t@t -c user.name=t add -A \
 && git -c user.email=t@t -c user.name=t commit -qm base
