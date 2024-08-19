# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
ENV ROBOFLOW_KEY=
# Expose the port that Flask will run on
EXPOSE 5000



# Define the command to run the Flask application
CMD ["python", "app.py"]
