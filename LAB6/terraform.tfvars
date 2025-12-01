# =============================================================================
# Terraform GCP Lab - Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# REQUIRED: Your GCP Project ID
# -----------------------------------------------------------------------------
project_id = "rjaditya-mlops-lab-479905"

# -----------------------------------------------------------------------------
# PHASE 1: MNIST Classifier v1 (GREEN theme)
# -----------------------------------------------------------------------------
#image_name      = "mnist-classifier"
#image_tag       = "latest"
#memory_limit    = "1Gi"
#cpu_limit       = "1"
#request_timeout = 300

# -----------------------------------------------------------------------------
# PHASE 2: MNIST Classifier v2 (DARK theme) - for REAPPLY demo!
# -----------------------------------------------------------------------------
image_name      = "mnist-classifier"
image_tag       = "v2"
memory_limit    = "1Gi"
cpu_limit       = "1"
request_timeout = 500

# -----------------------------------------------------------------------------
# Cost optimization (keep these!)
# -----------------------------------------------------------------------------
region          = "us-central1"
app_name        = "mnist-classifier"
environment     = "dev"
min_instances   = 0   # Scale to zero = FREE when idle!
max_instances   = 1   # Limit scaling

# Allow public access for testing
allow_public_access = true
