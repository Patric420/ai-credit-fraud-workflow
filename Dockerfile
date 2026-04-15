FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt

COPY streamlit_benchmark_dashboard.py .
COPY artifacts ./artifacts

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_benchmark_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
