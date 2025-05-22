# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependencies file first and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set default command to run your main script
CMD ["python", "main.py"]

# To run the regression example instead, use:
# docker run --rm ai-exercises python regression.py
