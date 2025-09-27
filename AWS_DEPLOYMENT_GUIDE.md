# AWS Deployment Guide for Trader AI

This guide will help you deploy your Trader AI application on AWS EC2.

## Prerequisites

1. **AWS EC2 Instance** (t3.medium or larger recommended)
2. **RDS PostgreSQL Database** (db.t3.micro or larger)
3. **ElastiCache Redis** (cache.t3.micro or larger)
4. **Security Groups** configured properly
5. **Domain name** (optional, can use IP address)

## Quick Start

### 1. Connect to your EC2 instance

```bash
ssh -i your-key.pem ec2-user@your-ec2-ip
```

### 2. Clone and setup the repository

```bash
# Clone your repository
git clone https://github.com/your-username/Trader_AI_V_0.1.git
cd Trader_AI_V_0.1

# Make scripts executable
chmod +x deploy-aws.sh configure-aws.sh monitor-aws.sh
```

### 3. Configure environment variables

```bash
# Run the configuration script
./configure-aws.sh
```

This script will ask you for:
- RDS database endpoint and password
- ElastiCache Redis endpoint
- AWS credentials
- Domain name (or use IP address)
- Optional: Angel One API, WhatsApp, Twilio credentials

### 4. Deploy the application

```bash
# Run the deployment script
./deploy-aws.sh
```

### 5. Monitor the application

```bash
# Check status
./monitor-aws.sh status

# View logs
./monitor-aws.sh logs app

# Check health
./monitor-aws.sh health
```

## Manual Configuration

If you prefer to configure manually, follow these steps:

### 1. Update environment variables

Edit the `.env` file with your AWS details:

```bash
# Database Configuration
DATABASE_URL=postgresql://trader_user:your_password@your-rds-endpoint:5432/trader_ai
REDIS_URL=redis://your-elasticache-endpoint:6379

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Application URLs (replace with your domain or IP)
ALLOWED_ORIGINS=http://your-domain.com,https://your-domain.com
REACT_APP_API_URL=http://your-domain.com/api
REACT_APP_WS_URL=ws://your-domain.com/ws

# Security
SECRET_KEY=your-secret-key-here
```

### 2. Install dependencies

```bash
# Update system
sudo yum update -y

# Install required packages
sudo yum install -y docker git python3 python3-pip nodejs npm nginx

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user
```

### 3. Deploy with Docker Compose

```bash
# Build and start services
docker-compose -f docker-compose.aws.yml up -d

# Check status
docker-compose -f docker-compose.aws.yml ps
```

### 4. Configure Nginx

```bash
# Copy nginx configuration
sudo cp frontend/nginx.conf /etc/nginx/conf.d/trader-ai.conf

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Security Group Configuration

Ensure your EC2 security group allows:

- **Port 22** (SSH) - Your IP only
- **Port 80** (HTTP) - 0.0.0.0/0
- **Port 443** (HTTPS) - 0.0.0.0/0 (if using SSL)
- **Port 8000** (API) - 0.0.0.0/0 (or restrict to load balancer)

## RDS Configuration

1. Create a PostgreSQL RDS instance
2. Configure security group to allow access from EC2
3. Note the endpoint and set up database user
4. Update `DATABASE_URL` in your `.env` file

## ElastiCache Configuration

1. Create a Redis ElastiCache cluster
2. Configure security group to allow access from EC2
3. Note the endpoint and update `REDIS_URL` in your `.env` file

## SSL/HTTPS Setup (Optional)

### Using Let's Encrypt (Free)

```bash
# Install Certbot
sudo yum install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Using AWS Certificate Manager

1. Request a certificate in AWS Certificate Manager
2. Validate domain ownership
3. Update nginx configuration to use SSL
4. Update environment variables to use HTTPS

## Monitoring and Maintenance

### Check Application Status

```bash
# Service status
./monitor-aws.sh status

# Application health
./monitor-aws.sh health

# View logs
./monitor-aws.sh logs app
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./monitor-aws.sh update
```

### Backup Data

```bash
# Create backup
./monitor-aws.sh backup
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check RDS security group
   - Verify database endpoint and credentials
   - Ensure database is running

2. **Redis Connection Failed**
   - Check ElastiCache security group
   - Verify Redis endpoint
   - Ensure Redis cluster is running

3. **Application Not Accessible**
   - Check EC2 security group
   - Verify nginx is running
   - Check application logs

4. **WebSocket Connection Failed**
   - Check nginx WebSocket configuration
   - Verify CORS settings
   - Check firewall rules

### Log Locations

- **Application logs**: `sudo journalctl -u trader-ai -f`
- **Docker logs**: `docker-compose -f docker-compose.aws.yml logs -f`
- **Nginx logs**: `sudo tail -f /var/log/nginx/access.log`

### Performance Optimization

1. **Database Optimization**
   - Enable connection pooling
   - Monitor query performance
   - Consider read replicas for heavy read workloads

2. **Redis Optimization**
   - Configure appropriate memory settings
   - Monitor cache hit rates
   - Use Redis clustering for high availability

3. **Application Optimization**
   - Enable gzip compression
   - Configure caching headers
   - Monitor resource usage

## Production Checklist

- [ ] RDS database configured and accessible
- [ ] ElastiCache Redis configured and accessible
- [ ] Security groups properly configured
- [ ] Environment variables set correctly
- [ ] SSL certificate installed (if using domain)
- [ ] Monitoring and logging configured
- [ ] Backup strategy implemented
- [ ] Application health checks working
- [ ] Performance monitoring in place

## Support

For issues or questions:
1. Check the logs using `./monitor-aws.sh logs`
2. Verify all services are running with `./monitor-aws.sh status`
3. Check the health endpoint at `http://your-domain/health`
4. Review this guide for common solutions

## Next Steps

After successful deployment:
1. Set up monitoring alerts
2. Configure automated backups
3. Implement CI/CD pipeline
4. Set up load balancing for high availability
5. Configure auto-scaling if needed
