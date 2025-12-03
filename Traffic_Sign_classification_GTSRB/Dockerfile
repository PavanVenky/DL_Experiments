# Use a minimal official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first (to use Docker layer caching)
COPY requirements.txt .

# Install dependencies system-wide inside container (no venv needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run your application (you can modify this)
CMD ["python", "test.py"]

LABEL name="traffic_sign_classifier"
