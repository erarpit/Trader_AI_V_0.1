# Trader AI - AWS Deployment Guide

This guide will help you deploy the Trader AI application on AWS infrastructure.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Terraform** installed (version 1.0+)
4. **Docker** and **Docker Compose** installed on EC2
5. **Git** installed on EC2

## Quick Start

### 1. Launch EC2 Instance

Launch an EC2 instance with the following specifications:
- **Instance Type**: t3.medium or larger
- **AMI**: Amazon Linux 2 or Ubuntu 20.04+
- **Storage**: 20GB+ EBS volume
- **Security Group**: Allow ports 22, 80, 443, 8000

### 2. Connect to EC2 Instance

```bash
ssh -i your-key.pem ec2-user@your-ec2-ip
```

### 3. Clone Repository

```bash
git clone <your-repo-url> Trader_AI_V_0.1
cd Trader_AI_V_0.1
```

### 4. Run AWS Setup

```bash
chmod +x aws-setup.sh
./aws-setup.sh
```

This script will:
- Create AWS infrastructure (VPC, RDS, ElastiCache, Security Groups)
- Configure environment variables
- Deploy the application using Docker

## Manual Setup

If you prefer to set up manually:

### 1. Create Infrastructure

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the infrastructure
terraform apply
```

### 2. Configure Environment

Create `.env.aws` file with your configuration:

```bash
cp env.aws .env.aws
# Edit .env.aws with your specific values
```

### 3. Deploy Application

```bash
chmod +x aws-deploy.sh
./aws-deploy.sh
```

## Infrastructure Components

The AWS setup creates:

### Networking
- **VPC** with public and private subnets
- **Internet Gateway** for public access
- **Route Tables** for traffic routing
- **Security Groups** for access control

### Database
- **RDS PostgreSQL** instance (db.t3.micro)
- **ElastiCache Redis** cluster (cache.t3.micro)
- **Subnet Groups** for database isolation

### Security
- **Security Groups** with minimal required access
- **Private Subnets** for databases
- **Public Subnets** for application servers

## Environment Variables

Key environment variables you need to configure:

```bash
# Database
DATABASE_URL=postgresql://trader_user:password@rds-endpoint:5432/trader_ai
REDIS_URL=redis://elasticache-endpoint:6379

# Application
SECRET_KEY=your-secret-key
DEBUG=False
HOST=0.0.0.0
PORT=8000

# CORS
ALLOWED_ORIGINS=http://your-domain.com,https://your-domain.com

# Frontend
REACT_APP_API_URL=http://your-ec2-ip:8000
REACT_APP_WS_URL=ws://your-ec2-ip:8000
```

## Production Considerations

### 1. SSL/TLS Certificate

For production, set up SSL certificates:

```bash
# Using Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 2. Domain Configuration

Update your domain's DNS to point to the EC2 instance's public IP.

### 3. Load Balancer

For high availability, consider using an Application Load Balancer:

```bash
# Add to terraform configuration
resource "aws_lb" "trader_ai_alb" {
  name               = "trader-ai-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.public_subnet_1.id, aws_subnet.public_subnet_2.id]
}
```

### 4. Auto Scaling

Set up Auto Scaling Groups for high availability:

```bash
# Add to terraform configuration
resource "aws_autoscaling_group" "trader_ai_asg" {
  name                = "trader-ai-asg"
  vpc_zone_identifier = [aws_subnet.public_subnet_1.id, aws_subnet.public_subnet_2.id]
  target_group_arns   = [aws_lb_target_group.trader_ai_tg.arn]
  health_check_type   = "ELB"
  min_size            = 1
  max_size            = 3
  desired_capacity    = 2
}
```

## Monitoring and Logging

### 1. CloudWatch Logs

```bash
# Install CloudWatch agent
sudo yum install -y amazon-cloudwatch-agent

# Configure logging
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 2. Application Monitoring

The application includes health check endpoints:
- Backend: `http://your-ec2-ip:8000/health`
- Frontend: `http://your-ec2-ip/`

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check security group rules
   - Verify RDS endpoint
   - Check database credentials

2. **CORS Errors**
   - Update `ALLOWED_ORIGINS` environment variable
   - Check frontend URL configuration

3. **WebSocket Connection Failed**
   - Verify WebSocket proxy configuration in nginx
   - Check security group rules for port 8000

### Logs

```bash
# View application logs
sudo docker-compose -f docker-compose.aws.yml logs -f

# View specific service logs
sudo docker-compose -f docker-compose.aws.yml logs -f backend
sudo docker-compose -f docker-compose.aws.yml logs -f frontend
```

### Health Checks

```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost

# Check database connection
sudo docker-compose -f docker-compose.aws.yml exec backend python -c "from core.database import engine; print('DB OK' if engine else 'DB Error')"
```

## Cost Optimization

### 1. Instance Sizing
- Start with t3.micro for development
- Scale up based on usage patterns
- Use Spot Instances for non-critical workloads

### 2. Database Optimization
- Use RDS Reserved Instances for production
- Enable automated backups
- Monitor storage usage

### 3. Monitoring Costs
- Set up billing alerts
- Use AWS Cost Explorer
- Review and optimize resource usage regularly

## Security Best Practices

1. **Network Security**
   - Use private subnets for databases
   - Implement security groups with minimal access
   - Enable VPC Flow Logs

2. **Application Security**
   - Use strong passwords and secrets
   - Enable encryption at rest and in transit
   - Regular security updates

3. **Access Control**
   - Use IAM roles instead of access keys
   - Implement least privilege access
   - Enable MFA for AWS console

## Backup and Recovery

### 1. Database Backups
- RDS automated backups (7 days retention)
- Manual snapshots before major changes
- Cross-region backup replication

### 2. Application Backups
- Code repository (Git)
- Configuration files
- Docker images

### 3. Disaster Recovery
- Multi-AZ deployment for RDS
- Cross-region replication
- Regular disaster recovery testing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Check AWS CloudWatch logs
4. Create an issue in the repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.
