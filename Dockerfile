FROM python:3.12-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
COPY src/ ./src/
COPY config.yaml feature_spec.json api_light.py app.py ./
USER appuser
ENV PORT=7860 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api_light:app --host 0.0.0.0 --port ${PORT:-7860}"]
