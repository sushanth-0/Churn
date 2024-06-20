FROM python:3.8.5-slim-buster

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install cmake and other required build tools
RUN apt-get update && apt-get install -y cmake

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install -r requirements.txt

# Set the entry point for the container
CMD ["python3", "app.py"]
