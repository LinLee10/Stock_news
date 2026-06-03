# Variables for Terraform configuration

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "stonk-news"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "stonk-news-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# Database
variable "create_rds" {
  description = "Whether to create RDS instance"
  type        = bool
  default     = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "RDS max allocated storage (GB)"
  type        = number
  default     = 100
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "stonk_news"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "stonk_admin"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "changeme123!"
}

variable "db_backup_retention_period" {
  description = "Database backup retention period (days)"
  type        = number
  default     = 7
}

# Redis
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
  default     = "changeme-redis-auth-token-123"
}

# Application secrets
variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key"
  type        = string
  sensitive   = true
  default     = "demo"
}

variable "finnhub_api_key" {
  description = "Finnhub API key"
  type        = string
  sensitive   = true
  default     = "demo"
}

variable "telegram_bot_token" {
  description = "Telegram bot token"
  type        = string
  sensitive   = true
  default     = "demo"
}

variable "email_password" {
  description = "Email password for notifications"
  type        = string
  sensitive   = true
  default     = "demo"
}

variable "jwt_secret_key" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
  default     = "your-super-secret-jwt-key-change-in-production"
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

# Logging
variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 30
}

# Environment-specific configurations
variable "node_instance_types" {
  description = "EC2 instance types for worker nodes by environment"
  type        = map(list(string))
  default = {
    dev = ["t3.small", "t3.medium"]
    staging = ["t3.medium", "t3.large"]
    production = ["t3.large", "t3.xlarge"]
  }
}

variable "node_desired_size" {
  description = "Desired number of worker nodes by environment"
  type        = map(number)
  default = {
    dev = 2
    staging = 3
    production = 5
  }
}

variable "node_max_size" {
  description = "Maximum number of worker nodes by environment"
  type        = map(number)
  default = {
    dev = 4
    staging = 8
    production = 20
  }
}

variable "node_min_size" {
  description = "Minimum number of worker nodes by environment"
  type        = map(number)
  default = {
    dev = 1
    staging = 2
    production = 3
  }
}

# Monitoring
variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

# Backup and disaster recovery
variable "backup_retention_period" {
  description = "Backup retention period for production"
  type        = number
  default     = 30
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for production"
  type        = bool
  default     = true
}

# Auto-scaling
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

# Security
variable "enable_security_groups" {
  description = "Enable additional security groups"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

# Cost optimization
variable "spot_instance_percentage" {
  description = "Percentage of spot instances in node groups"
  type        = number
  default     = 50
}

variable "enable_cost_allocation_tags" {
  description = "Enable cost allocation tags"
  type        = bool
  default     = true
}

# Development
variable "enable_ssh_access" {
  description = "Enable SSH access to worker nodes (dev only)"
  type        = bool
  default     = false
}

variable "ssh_key_name" {
  description = "EC2 Key Pair name for SSH access"
  type        = string
  default     = ""
}

# Local variables for environment-specific settings
locals {
  environment_config = {
    dev = {
      instance_types = ["t3.small", "t3.medium"]
      desired_size   = 2
      max_size       = 4
      min_size       = 1
      disk_size      = 20
      capacity_type  = "SPOT"
    }
    staging = {
      instance_types = ["t3.medium", "t3.large"]
      desired_size   = 3
      max_size       = 8
      min_size       = 2
      disk_size      = 30
      capacity_type  = "ON_DEMAND"
    }
    production = {
      instance_types = ["t3.large", "t3.xlarge", "t3.2xlarge"]
      desired_size   = 5
      max_size       = 20
      min_size       = 3
      disk_size      = 50
      capacity_type  = "ON_DEMAND"
    }
  }
  
  # Common tags
  common_tags = {
    Environment   = var.environment
    Project       = var.project_name
    ManagedBy     = "terraform"
    Owner         = "devops-team"
    CostCenter    = "engineering"
    BackupPolicy  = var.environment == "production" ? "daily" : "weekly"
    Compliance    = var.environment == "production" ? "required" : "optional"
  }
  
  # Environment-specific resource naming
  resource_prefix = "${var.project_name}-${var.environment}"
  
  # SSL/TLS configuration
  enable_ssl = var.domain_name != ""
  
  # High availability configuration
  enable_ha = var.environment == "production"
  multi_az  = var.environment == "production"
  
  # Security configuration
  encryption_enabled = var.environment != "dev"
  
  # Monitoring configuration
  detailed_monitoring = var.environment == "production"
  
  # Backup configuration
  backup_enabled = var.environment != "dev"
}