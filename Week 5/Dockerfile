FROM python:3.12-slim

WORKDIR /code

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Retrain inside the image so the saved model matches the installed scikit-learn
RUN python -m training.train_model

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
