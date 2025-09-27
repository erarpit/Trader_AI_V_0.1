# AWS Infrastructure for Trader AI
# This Terraform configuration creates the necessary AWS resources

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# VPC
resource "aws_vpc" "trader_ai_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "trader-ai-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "trader_ai_igw" {
  vpc_id = aws_vpc.trader_ai_vpc.id

  tags = {
    Name = "trader-ai-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public_subnet_1" {
  vpc_id                  = aws_vpc.trader_ai_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "trader-ai-public-subnet-1"
    Environment = var.environment
  }
}

resource "aws_subnet" "public_subnet_2" {
  vpc_id                  = aws_vpc.trader_ai_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = {
    Name = "trader-ai-public-subnet-2"
    Environment = var.environment
  }
}

# Private Subnets
resource "aws_subnet" "private_subnet_1" {
  vpc_id            = aws_vpc.trader_ai_vpc.id
  cidr_block        = "10.0.3.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "trader-ai-private-subnet-1"
    Environment = var.environment
  }
}

resource "aws_subnet" "private_subnet_2" {
  vpc_id            = aws_vpc.trader_ai_vpc.id
  cidr_block        = "10.0.4.0/24"
  availability_zone = "${var.aws_region}b"

  tags = {
    Name = "trader-ai-private-subnet-2"
    Environment = var.environment
  }
}

# Route Table for Public Subnets
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.trader_ai_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.trader_ai_igw.id
  }

  tags = {
    Name = "trader-ai-public-rt"
    Environment = var.environment
  }
}

# Route Table Associations
resource "aws_route_table_association" "public_subnet_1_association" {
  subnet_id      = aws_subnet.public_subnet_1.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "public_subnet_2_association" {
  subnet_id      = aws_subnet.public_subnet_2.id
  route_table_id = aws_route_table.public_rt.id
}

# Security Group for EC2
resource "aws_security_group" "ec2_sg" {
  name_prefix = "trader-ai-ec2-sg"
  vpc_id      = aws_vpc.trader_ai_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP access
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS access
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # API access
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "trader-ai-ec2-sg"
    Environment = var.environment
  }
}

# Security Group for RDS
resource "aws_security_group" "rds_sg" {
  name_prefix = "trader-ai-rds-sg"
  vpc_id      = aws_vpc.trader_ai_vpc.id

  # PostgreSQL access from EC2
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }

  tags = {
    Name = "trader-ai-rds-sg"
    Environment = var.environment
  }
}

# Security Group for ElastiCache
resource "aws_security_group" "elasticache_sg" {
  name_prefix = "trader-ai-elasticache-sg"
  vpc_id      = aws_vpc.trader_ai_vpc.id

  # Redis access from EC2
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }

  tags = {
    Name = "trader-ai-elasticache-sg"
    Environment = var.environment
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "trader_ai_db_subnet_group" {
  name       = "trader-ai-db-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]

  tags = {
    Name = "trader-ai-db-subnet-group"
    Environment = var.environment
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "trader_ai_db" {
  identifier = "trader-ai-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "trader_ai"
  username = "trader_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.trader_ai_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name = "trader-ai-db"
    Environment = var.environment
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "trader_ai_cache_subnet_group" {
  name       = "trader-ai-cache-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "trader_ai_redis" {
  replication_group_id       = "trader-ai-redis"
  description                = "Redis cluster for Trader AI"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 1
  
  subnet_group_name          = aws_elasticache_subnet_group.trader_ai_cache_subnet_group.name
  security_group_ids         = [aws_security_group.elasticache_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = false
  
  tags = {
    Name = "trader-ai-redis"
    Environment = var.environment
  }
}

# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.trader_ai_vpc.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = [aws_subnet.public_subnet_1.id, aws_subnet.public_subnet_2.id]
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
}

output "ec2_security_group_id" {
  description = "ID of the EC2 security group"
  value       = aws_security_group.ec2_sg.id
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.trader_ai_db.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.trader_ai_redis.primary_endpoint_address
}

# Variables that need to be set
variable "db_password" {
  description = "Password for the RDS database"
  type        = string
  sensitive   = true
}
