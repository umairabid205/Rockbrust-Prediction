#!/bin/bash
# Safe startup script for Docker Compose services
# This prevents resource spikes by starting services in stages

set -e

echo "🚀 Starting Docker Compose services safely with resource monitoring..."

# Function to check CPU usage
check_cpu_usage() {
    local threshold=80
    local cpu_usage=$(docker stats --no-stream --format "table {{.CPUPerc}}" | tail -n +2 | sed 's/%//' | awk '{sum+=$1} END {print int(sum)}')
    
    if [ "$cpu_usage" -gt "$threshold" ]; then
        echo "⚠️  High CPU usage detected: ${cpu_usage}%. Waiting for resources to free up..."
        return 1
    fi
    return 0
}

# Function to wait for service health
wait_for_healthy() {
    local service=$1
    local max_wait=60
    local count=0
    
    echo "⏳ Waiting for $service to be healthy..."
    
    while [ $count -lt $max_wait ]; do
        if docker-compose ps $service | grep -q "healthy"; then
            echo "✅ $service is healthy"
            return 0
        fi
        
        if docker-compose ps $service | grep -q "exited\|restarting"; then
            echo "❌ $service failed to start properly"
            docker-compose logs --tail 20 $service
            return 1
        fi
        
        sleep 2
        count=$((count + 1))
    done
    
    echo "⚠️  Timeout waiting for $service to be healthy"
    return 1
}

echo "📋 Step 1: Starting core infrastructure (PostgreSQL, Redis)..."
docker-compose up -d postgres redis

wait_for_healthy postgres
wait_for_healthy redis

echo "📋 Step 2: Initializing Airflow database..."
docker-compose up airflow-init
wait_for_healthy airflow-init || true  # init containers exit after completion

echo "📋 Step 3: Starting Airflow core services..."
docker-compose up -d airflow-scheduler airflow-apiserver

wait_for_healthy airflow-scheduler
wait_for_healthy airflow-apiserver

echo "📋 Step 4: Starting Airflow workers..."
docker-compose up -d airflow-worker airflow-triggerer airflow-dag-processor

wait_for_healthy airflow-worker
wait_for_healthy airflow-triggerer
wait_for_healthy airflow-dag-processor

echo "📋 Step 5: Starting MLflow services..."
docker-compose up -d mlflow-minio

wait_for_healthy mlflow-minio

echo "📋 Step 6: Starting MLflow server..."
docker-compose up -d mlflow-server

wait_for_healthy mlflow-server

echo "🎉 All services started successfully!"
echo ""
echo "🌐 Access your services:"
echo "   • Airflow UI: http://localhost:8080 (airflow/airflow)"
echo "   • MLflow UI:  http://localhost:5001"
echo "   • MinIO Console: http://localhost:9001 (minio_access_key/minio_secret_key)"
echo ""
echo "📊 Monitoring services..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Final CPU check
echo ""
echo "🔍 Final system resource check..."
if check_cpu_usage; then
    echo "✅ System resources are within normal limits"
else
    echo "⚠️  High CPU usage detected. Consider stopping some services if issues persist."
    echo "   Use: docker-compose down [service_name]"
fi
