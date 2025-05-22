# AI Exercises

This project is for practicing and experimenting with AI and machine learning models using Python.

## Folder Structure

- `main.py` – Main script to run experiments.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – (Optional) For containerized development.
- `venv/` – Virtual environment folder (not included in Git).
- Other `.py` files – Experiment scripts and utilities.

## Getting Started

### 1. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows

```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the regression example
```bash
python regression.py
```

## Scripts

### regression.py
Trains a linear regression model using scikit-learn on synthetic data, prints the model's intercept, coefficient, and R^2 score.

## Docker Usage
To run the project in a Docker container:
```bash
docker build -t ai-exercises .
docker run --rm ai-exercises
```
