#!/bin/bash
# MLflow MinIO Configuration Script
# Run this before testing MLflow: source ./configure_minio.sh

echo "🔧 Configuring MLflow with MinIO credentials..."

# Set MinIO credentials for MLflow (matches docker-compose.yaml)
export AWS_ACCESS_KEY_ID="minio_access_key"
export AWS_SECRET_ACCESS_KEY="minio_secret_key" 
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9001"  # MinIO API port

echo "✅ MinIO credentials configured for MLflow"
echo "📊 MLflow UI: http://localhost:5001"
echo "📦 MinIO Console: http://localhost:9091"

echo "🪣 Creating required MinIO bucket..."
# Try to create the mlflow bucket using MinIO client commands
docker exec mlflow_minio mc config host add local http://localhost:9000 minio_access_key minio_secret_key 2>/dev/null || true
docker exec mlflow_minio mc mb local/mlflow 2>/dev/null || echo "   Bucket 'mlflow' may already exist"

echo "🚀 Now run: python test_mlflow.py"
echo ""
echo "🔍 If model logging still fails, run this test:"
echo "   AWS_ACCESS_KEY_ID=\"minio_access_key\" AWS_SECRET_ACCESS_KEY=\"minio_secret_key\" MLFLOW_S3_ENDPOINT_URL=\"http://localhost:9001\" python test_mlflow_simple.py"
