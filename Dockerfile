# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code and model files to the container
COPY . /app

# Copy the models directory into the container (Add this line)
COPY models /app/models

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Flask
ENV FLASK_APP=src/web_app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port for Flask application
EXPOSE 5000

# Run the Flask app with host set to 0.0.0.0
CMD ["flask", "run", "--host=0.0.0.0"]
