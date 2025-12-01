# =============================================================================
# Terraform GCP Lab - Variable Definitions
# =============================================================================
# Optimized for COST SAVINGS - cheapest Cloud Run configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Required Variables
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "The GCP project ID"
  type        = string

  validation {
    condition     = length(var.project_id) > 0
    error_message = "Project ID cannot be empty."
  }
}

variable "image_name" {
  description = "Name of the Docker image (without registry path)"
  type        = string
  default     = "mnist-classifier"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

# -----------------------------------------------------------------------------
# Optional Variables with Cost-Optimized Defaults
# -----------------------------------------------------------------------------

variable "region" {
  description = "GCP region (us-central1 is often cheapest)"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Name of the application"
  type        = string
  default     = "mnist-classifier"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,48}[a-z0-9]$", var.app_name))
    error_message = "App name must be lowercase, start with letter, max 50 chars."
  }
}

variable "environment" {
  description = "Environment label"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be: dev, staging, or prod."
  }
}

# -----------------------------------------------------------------------------
# Cloud Run Scaling (COST CRITICAL!)
# -----------------------------------------------------------------------------

variable "min_instances" {
  description = "Minimum instances (0 = scale to zero for FREE when idle!)"
  type        = number
  default     = 0  # CHEAPEST: No cost when no traffic!

  validation {
    condition     = var.min_instances >= 0 && var.min_instances <= 10
    error_message = "Min instances must be 0-10. Use 0 for lowest cost."
  }
}

variable "max_instances" {
  description = "Maximum instances (limit to control costs)"
  type        = number
  default     = 1  # Limit to 1 for cost control

  validation {
    condition     = var.max_instances >= 1 && var.max_instances <= 10
    error_message = "Max instances must be 1-10."
  }
}

# -----------------------------------------------------------------------------
# Cloud Run Resources (COST CRITICAL!)
# -----------------------------------------------------------------------------

variable "cpu_limit" {
  description = "CPU limit (1 = 1 vCPU). Lower = cheaper"
  type        = string
  default     = "1"  # Minimum for TensorFlow

  validation {
    condition     = contains(["1", "2", "4"], var.cpu_limit)
    error_message = "CPU must be: 1, 2, or 4."
  }
}

variable "memory_limit" {
  description = "Memory limit. MNIST: 1Gi, VLM: 4-8Gi"
  type        = string
  default     = "1Gi"  # Minimum for TensorFlow model loading

  validation {
    condition     = contains(["512Mi", "1Gi", "2Gi", "4Gi", "8Gi"], var.memory_limit)
    error_message = "Memory must be: 512Mi, 1Gi, 2Gi, 4Gi, or 8Gi."
  }
}

variable "request_timeout" {
  description = "Request timeout in seconds (MNIST: 300s, VLM: 900s for model download)"
  type        = number
  default     = 300  # TensorFlow model loading can take time on cold start

  validation {
    condition     = var.request_timeout >= 60 && var.request_timeout <= 3600
    error_message = "Timeout must be 60-3600 seconds."
  }
}

# -----------------------------------------------------------------------------
# Access Control
# -----------------------------------------------------------------------------

variable "allow_public_access" {
  description = "Allow unauthenticated public access"
  type        = bool
  default     = true  # Set to true for easy testing
}
