FROM python:3.12.2 AS builder
 
WORKDIR /app
 
RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
 
COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip install flask
RUN pip install torch torchaudio torchvision flask Gunicorn speechbrain==0.5.16 transformers soundfile

# Stage 2
FROM python:3.12.2 AS runner
 
WORKDIR /app
 
COPY --from=builder /app/venv venv
COPY app.py app.py
 
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV FLASK_APP=app/app.py
 
EXPOSE 8080
 
CMD ["gunicorn", "--bind" , ":8080", "--workers", "2", "app:app"]
