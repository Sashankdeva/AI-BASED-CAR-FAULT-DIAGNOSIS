FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train at build time if no model was committed, so the image is self-contained.
RUN test -f models/fault_model.joblib || (python -m car_fault.train && python -m car_fault.evaluate)

ENV CARFAULT_CORS_ORIGINS="*"
EXPOSE 8005
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8005"]
