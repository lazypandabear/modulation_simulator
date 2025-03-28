# Use an official lightweight Python image.
FROM python:3.13-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy and install dependencies.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose the port your Flask app will run on.
EXPOSE 5000

# Set environment variables for Flask.
ENV FLASK_APP=sims.py
ENV FLASK_RUN_HOST=0.0.0.0

# Install gunicorn
RUN pip install gunicorn

# Run the Flask application using gunicorn.
CMD ["gunicorn", "--workers", "3", "--timeout", "120", "--bind", "0.0.0.0:5000", "sims:app"]
