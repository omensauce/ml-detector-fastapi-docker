FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Heavy deps in their own layer so code changes don't bust this cache.
# numpy must precede torch (torch wheels embed a copy but pip may resolve differently).
RUN pip install --upgrade pip && \
    pip install "numpy>=2,<3" && \
    pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Model artifact first — changes less often than app code
COPY fruits_model.pt .
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
