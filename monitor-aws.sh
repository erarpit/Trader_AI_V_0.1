#!/bin/bash

# AWS Monitoring Script for Trader AI
# This script provides monitoring and maintenance functions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check service status
check_services() {
    print_header "Service Status"
    
    # Check Trader AI service
    if systemctl is-active --quiet trader-ai; then
        print_status "✅ Trader AI service is running"
    else
        print_error "❌ Trader AI service is not running"
    fi
    
    # Check Nginx
    if systemctl is-active --quiet nginx; then
        print_status "✅ Nginx is running"
    else
        print_error "❌ Nginx is not running"
    fi
    
    # Check Docker containers
    print_header "Docker Containers"
    docker-compose -f docker-compose.aws.yml ps
    
    # Check disk space
    print_header "Disk Usage"
    df -h
    
    # Check memory usage
    print_header "Memory Usage"
    free -h
    
    # Check CPU usage
    print_header "CPU Usage"
    top -bn1 | grep "Cpu(s)"
}

# Function to view logs
view_logs() {
    local service=$1
    
    case $service in
        "app"|"trader-ai")
            print_header "Trader AI Application Logs"
            sudo journalctl -u trader-ai -f --lines=50
            ;;
        "docker")
            print_header "Docker Container Logs"
            docker-compose -f docker-compose.aws.yml logs -f --tail=50
            ;;
        "nginx")
            print_header "Nginx Logs"
            sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log
            ;;
        "backend")
            print_header "Backend Container Logs"
            docker-compose -f docker-compose.aws.yml logs -f backend --tail=50
            ;;
        "frontend")
            print_header "Frontend Container Logs"
            docker-compose -f docker-compose.aws.yml logs -f frontend --tail=50
            ;;
        *)
            print_error "Unknown service: $service"
            echo "Available services: app, docker, nginx, backend, frontend"
            ;;
    esac
}

# Function to restart services
restart_services() {
    local service=$1
    
    case $service in
        "all")
            print_header "Restarting All Services"
            sudo systemctl restart trader-ai
            sudo systemctl restart nginx
            ;;
        "app"|"trader-ai")
            print_header "Restarting Trader AI Service"
            sudo systemctl restart trader-ai
            ;;
        "nginx")
            print_header "Restarting Nginx"
            sudo systemctl restart nginx
            ;;
        "docker")
            print_header "Restarting Docker Containers"
            docker-compose -f docker-compose.aws.yml restart
            ;;
        *)
            print_error "Unknown service: $service"
            echo "Available services: all, app, nginx, docker"
            ;;
    esac
}

# Function to check application health
check_health() {
    print_header "Application Health Check"
    
    local public_ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
    
    echo "Checking health endpoint..."
    if curl -f -s "http://localhost/health" > /dev/null; then
        print_status "✅ Health check passed"
        curl -s "http://localhost/health" | jq '.' 2>/dev/null || curl -s "http://localhost/health"
    else
        print_error "❌ Health check failed"
    fi
    
    echo ""
    echo "Checking API endpoints..."
    if curl -f -s "http://localhost/api/realtime/market-status" > /dev/null; then
        print_status "✅ API endpoints are responding"
    else
        print_error "❌ API endpoints are not responding"
    fi
    
    echo ""
    echo "Application URLs:"
    echo "  - Frontend: http://$public_ip"
    echo "  - API Docs: http://$public_ip/docs"
    echo "  - Health: http://$public_ip/health"
}

# Function to update application
update_app() {
    print_header "Updating Application"
    
    # Pull latest changes
    print_status "Pulling latest changes..."
    git pull origin main
    
    # Rebuild containers
    print_status "Rebuilding containers..."
    docker-compose -f docker-compose.aws.yml build --no-cache
    
    # Restart services
    print_status "Restarting services..."
    docker-compose -f docker-compose.aws.yml down
    docker-compose -f docker-compose.aws.yml up -d
    
    print_status "✅ Application updated successfully"
}

# Function to backup data
backup_data() {
    print_header "Creating Backup"
    
    local backup_dir="/opt/backups/trader_ai_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup environment file
    cp .env "$backup_dir/"
    
    # Backup database (if using local SQLite)
    if [ -f "trader_ai.db" ]; then
        cp trader_ai.db "$backup_dir/"
    fi
    
    # Backup logs
    sudo journalctl -u trader-ai --since "1 day ago" > "$backup_dir/trader_ai.log"
    
    print_status "✅ Backup created at: $backup_dir"
}

# Function to show system information
show_info() {
    print_header "System Information"
    
    echo "Server Details:"
    echo "  - Hostname: $(hostname)"
    echo "  - OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    echo "  - Kernel: $(uname -r)"
    echo "  - Uptime: $(uptime -p)"
    
    echo ""
    echo "Network Information:"
    echo "  - Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'unknown')"
    echo "  - Private IP: $(hostname -I | awk '{print $1}')"
    
    echo ""
    echo "Application Information:"
    echo "  - Working Directory: $(pwd)"
    echo "  - Docker Version: $(docker --version)"
    echo "  - Docker Compose Version: $(docker-compose --version)"
}

# Main menu
show_menu() {
    echo ""
    print_header "Trader AI AWS Monitoring Tool"
    echo "1. Check service status"
    echo "2. View logs (app|docker|nginx|backend|frontend)"
    echo "3. Restart services (all|app|nginx|docker)"
    echo "4. Check application health"
    echo "5. Update application"
    echo "6. Create backup"
    echo "7. Show system information"
    echo "8. Exit"
    echo ""
}

# Main script logic
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
        read -p "Select an option (1-8): " choice
        
        case $choice in
            1)
                check_services
                ;;
            2)
                read -p "Enter service name: " service
                view_logs "$service"
                ;;
            3)
                read -p "Enter service name: " service
                restart_services "$service"
                ;;
            4)
                check_health
                ;;
            5)
                update_app
                ;;
            6)
                backup_data
                ;;
            7)
                show_info
                ;;
            8)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-8."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Command line mode
    case $1 in
        "status")
            check_services
            ;;
        "logs")
            view_logs "$2"
            ;;
        "restart")
            restart_services "$2"
            ;;
        "health")
            check_health
            ;;
        "update")
            update_app
            ;;
        "backup")
            backup_data
            ;;
        "info")
            show_info
            ;;
        *)
            echo "Usage: $0 [status|logs|restart|health|update|backup|info] [service]"
            echo "Or run without arguments for interactive mode"
            exit 1
            ;;
    esac
fi
