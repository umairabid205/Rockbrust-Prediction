#!/bin/bash
# Resource monitoring script for Docker containers
# Helps identify high CPU/Memory usage containers

echo "🔍 Docker Container Resource Monitor"
echo "=================================="
echo ""

# Function to get container stats with color coding
monitor_resources() {
    echo "📊 Current Resource Usage:"
    echo "Container Name                     CPU %    Memory Usage    Memory %    Status"
    echo "--------------------------------------------------------------------------------"
    
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.Container}}" | tail -n +2 | while read line; do
        name=$(echo $line | awk '{print $1}')
        cpu=$(echo $line | awk '{print $2}' | sed 's/%//')
        mem_usage=$(echo $line | awk '{print $3}')
        mem_percent=$(echo $line | awk '{print $4}' | sed 's/%//')
        container_id=$(echo $line | awk '{print $5}')
        
        # Get container status
        status=$(docker ps --format "{{.Status}}" --filter "id=$container_id")
        
        # Color coding for high usage
        if (( $(echo "$cpu > 100" | bc -l) )); then
            echo "🔴 $name: $cpu% CPU - HIGH USAGE! Memory: $mem_usage ($mem_percent%) - $status"
        elif (( $(echo "$cpu > 50" | bc -l) )); then
            echo "🟡 $name: $cpu% CPU - Medium usage. Memory: $mem_usage ($mem_percent%) - $status"
        else
            echo "🟢 $name: $cpu% CPU - Normal. Memory: $mem_usage ($mem_percent%) - $status"
        fi
    done
}

# Function to check for problematic containers
check_problem_containers() {
    echo ""
    echo "🚨 Checking for problematic containers..."
    
    problem_found=false
    
    docker stats --no-stream --format "{{.Name}} {{.CPUPerc}}" | while read name cpu; do
        cpu_num=$(echo $cpu | sed 's/%//')
        if (( $(echo "$cpu_num > 150" | bc -l) )); then
            echo "⚠️  HIGH CPU: $name is using $cpu CPU!"
            echo "   Consider restarting: docker-compose restart $name"
            problem_found=true
        fi
    done
    
    if [ "$problem_found" = false ]; then
        echo "✅ No problematic containers detected"
    fi
}

# Function to show service URLs
show_service_info() {
    echo ""
    echo "🌐 Service Access Information:"
    echo "-----------------------------"
    
    if docker ps | grep -q "airflow-apiserver"; then
        echo "✅ Airflow UI: http://localhost:8080 (airflow/airflow)"
    else
        echo "❌ Airflow UI: Not running"
    fi
    
    if docker ps | grep -q "mlflow_server"; then
        echo "✅ MLflow UI: http://localhost:5001"
    else
        echo "❌ MLflow UI: Not running"
    fi
    
    if docker ps | grep -q "mlflow_minio"; then
        echo "✅ MinIO Console: http://localhost:9001 (minio_access_key/minio_secret_key)"
    else
        echo "❌ MinIO Console: Not running"
    fi
}

# Function for quick actions
show_quick_actions() {
    echo ""
    echo "🛠️  Quick Actions:"
    echo "------------------"
    echo "• Monitor continuously: watch -n 5 './monitor.sh'"
    echo "• Stop all services: docker-compose down"
    echo "• Start services safely: ./start_services.sh"
    echo "• Restart a service: docker-compose restart [service-name]"
    echo "• View service logs: docker-compose logs [service-name]"
}

# Main execution
monitor_resources
check_problem_containers
show_service_info
show_quick_actions

echo ""
echo "💡 Tip: Run './monitor.sh' regularly to keep an eye on resource usage"
