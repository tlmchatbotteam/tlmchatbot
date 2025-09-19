FROM python:3.11-slim
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Flask port (matches app default PORT=8080)
EXPOSE 8080

# Start the app
CMD ["python", "rule-based.py"]
