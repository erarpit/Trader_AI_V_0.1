# Database Setup Guide

This guide explains how to configure the database for Trader AI.

## Database Options

### 1. SQLite (Development/Testing)
- **Pros**: Simple, no setup required, good for development
- **Cons**: Single-user, limited performance, not suitable for production
- **Use when**: Local development, testing, single-user applications

### 2. PostgreSQL (Production)
- **Pros**: Multi-user, high performance, ACID compliance, AWS RDS support
- **Cons**: Requires setup, more complex
- **Use when**: Production deployment, multiple users, AWS deployment

## Quick Setup

### Option 1: Use the setup script
```bash
chmod +x setup-database.sh
./setup-database.sh
```

### Option 2: Manual configuration

#### For SQLite (Development):
```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=sqlite:///./trader_ai.db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
DEBUG=True
EOF
```

#### For PostgreSQL (Production):
```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://username:password@host:port/database
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
DEBUG=False
EOF
```

## AWS Deployment

For AWS deployment, you have two options:

### Option 1: Use SQLite (Simple but Limited)
1. The app will use SQLite by default if no `DATABASE_URL` is set
2. Data will be stored in a Docker volume
3. **Not recommended for production** due to performance limitations

### Option 2: Use PostgreSQL with AWS RDS (Recommended)
1. Create an RDS PostgreSQL instance
2. Update `env.aws` with your RDS details:
   ```bash
   DATABASE_URL=postgresql://trader_user:your_password@your-rds-endpoint:5432/trader_ai
   ```
3. Run `./configure-aws.sh` to set up the environment

## Environment Files

- `.env` - Local development (not committed to git)
- `env.aws` - AWS production template (committed to git)
- `env.example` - Example configuration (committed to git)

## Database Migration

The application will automatically create tables when it starts. No manual migration is required.

## Troubleshooting

### SQLite Issues
- Check file permissions
- Ensure the directory is writable
- Check disk space

### PostgreSQL Issues
- Verify connection string
- Check if PostgreSQL is running
- Verify network connectivity
- Check firewall rules

### AWS RDS Issues
- Verify security groups
- Check RDS instance status
- Verify credentials
- Check VPC configuration

## Performance Considerations

### SQLite
- Good for up to ~1000 concurrent users
- Single writer limitation
- File-based, limited by disk I/O

### PostgreSQL
- Handles thousands of concurrent users
- Multiple writers
- Optimized for high-performance applications
- Better for trading applications with real-time data

## Recommendation

- **Development**: Use SQLite for simplicity
- **Production**: Use PostgreSQL with AWS RDS for reliability and performance
- **AWS Deployment**: PostgreSQL is the better choice for a trading application
