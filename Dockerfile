FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY diet_opt ./diet_opt
COPY data ./data
COPY *.json ./
RUN pip install --no-cache-dir ".[web]"
EXPOSE 8000
CMD ["uvicorn", "diet_opt.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
