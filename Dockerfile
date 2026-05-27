FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --timeout 1000 -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]