FROM python:3.10-slim

WORKDIR /app

# 1) Install Pipenv + deps
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv \
 && pipenv install --deploy --system --ignore-pipfile

# 2) Copy your code
COPY . .

# 3) Set HF cache to /app/cache (writable)
ENV HF_HOME=/app/cache/huggingface

# Expose and run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]















