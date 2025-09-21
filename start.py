#!/usr/bin/env python3
"""
Trader AI Startup Script
Starts the complete trading platform with all services
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def check_requirements():
    """Check if required tools are installed"""
    required_tools = ['python', 'node', 'npm']
    missing_tools = []
    
    for tool in required_tools:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install the missing tools and try again.")
        sys.exit(1)
    
    print("✅ All required tools are installed")

def check_python_packages():
    """Check if Python packages are installed"""
    try:
        import fastapi
        import uvicorn
        import redis
        import pandas
        import numpy
        print("✅ Python packages are installed")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        sys.exit(1)

def check_node_packages():
    """Check if Node packages are installed"""
    frontend_path = Path("frontend")
    if not (frontend_path / "node_modules").exists():
        print("❌ Node packages not installed")
        print("Please run: cd frontend && npm install")
        sys.exit(1)
    print("✅ Node packages are installed")

def start_redis():
    """Start Redis server"""
    print("🔄 Starting Redis server...")
    try:
        redis_process = subprocess.Popen(['redis-server'], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
        time.sleep(2)
        print("✅ Redis server started")
        return redis_process
    except FileNotFoundError:
        print("❌ Redis not found. Please install Redis and try again.")
        sys.exit(1)

def start_backend():
    """Start the backend API server"""
    print("🔄 Starting backend API server...")
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 
        'main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000',
        '--reload'
    ], cwd='backend')
    
    # Wait for backend to start
    time.sleep(5)
    print("✅ Backend API server started on http://localhost:8000")
    return backend_process

def start_frontend():
    """Start the frontend development server"""
    print("🔄 Starting frontend development server...")
    frontend_process = subprocess.Popen(['npm', 'start'], cwd='frontend')
    
    # Wait for frontend to start
    time.sleep(10)
    print("✅ Frontend development server started on http://localhost:3000")
    return frontend_process

def cleanup_processes(processes):
    """Clean up all started processes"""
    print("\n🔄 Shutting down services...")
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    print("✅ All services stopped")

def main():
    """Main startup function"""
    print("🚀 Starting Trader AI Platform...")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    check_python_packages()
    check_node_packages()
    
    processes = []
    
    try:
        # Start Redis
        redis_process = start_redis()
        processes.append(redis_process)
        
        # Start Backend
        backend_process = start_backend()
        processes.append(backend_process)
        
        # Start Frontend
        frontend_process = start_frontend()
        processes.append(frontend_process)
        
        print("\n" + "=" * 50)
        print("🎉 Trader AI Platform is running!")
        print("=" * 50)
        print("📊 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop all services")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
        cleanup_processes(processes)
        print("👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_processes(processes)
        sys.exit(1)

if __name__ == "__main__":
    main()
