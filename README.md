# MLOps-Project

This project implements an end-to-end MLOps pipeline for credit card fraud detection. The goal is to classify transactions as fraudulent or non-fraudulent while following MLOps best practices. The project covers data versioning, experiment tracking, model management, containerization, orchestration, and CI/CD automation. :contentReference[oaicite:0]{index=0}

## Dataset

The project uses the Credit Card Fraud Detection dataset containing European card transactions. The dataset is highly imbalanced and includes anonymized features obtained using PCA. :contentReference[oaicite:1]{index=1}

## Technologies Used

- Python
- DVC
- MLflow
- FastAPI
- Docker
- Kubernetes
- GitHub Actions

## Pipeline

1. **Data preprocessing**
   - Remove duplicates
   - Handle missing values
   - Scale features

2. **Data versioning**
   - Track datasets with DVC
   - Create reproducible pipelines

3. **Model training**
   - Train multiple models
   - Compare performance
   - Select the best model

4. **Experiment tracking**
   - Log parameters and metrics with MLflow
   - Save trained models

5. **Deployment**
   - Export the champion model
   - Serve predictions using FastAPI
   - Package the application with Docker

6. **Orchestration**
   - Deploy the Docker container on Kubernetes
   - Use deployment and service YAML files

7. **CI/CD**
   - Automate training and evaluation with GitHub Actions
   - Build and push the Docker image

## Project Structure

```
data/
src/
models/
api/
k8s/
.github/workflows/
Dockerfile
dvc.yaml
requirements.txt
README.md
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
dvc repro
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Build the Docker image:

```bash
docker build -t mlops-project .
```

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/
```

