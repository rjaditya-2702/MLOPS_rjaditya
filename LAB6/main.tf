# =============================================================================
# Terraform GCP Lab - MNIST Digit Classifier Deployment
# =============================================================================
# Deploys the LAB5 MNIST Flask app to Google Cloud Run
# Cloud Run: Pay only when requests are made, scales to zero when idle
# =============================================================================

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Provider Configuration
# -----------------------------------------------------------------------------
provider "google" {
  project = var.project_id
  region  = var.region
}

# -----------------------------------------------------------------------------
# Local Values - Construct image URL from variables
# -----------------------------------------------------------------------------
locals {
  container_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}-repo/${var.image_name}:${var.image_tag}"
}

# -----------------------------------------------------------------------------
# Enable Required APIs
# -----------------------------------------------------------------------------
resource "google_project_service" "run_api" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry_api" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# -----------------------------------------------------------------------------
# Artifact Registry - Docker Image Repository
# -----------------------------------------------------------------------------
resource "google_artifact_registry_repository" "mnist_repo" {
  location      = var.region
  repository_id = "${var.app_name}-repo"
  description   = "Docker repository for MNIST classifier"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    lab         = "lab6"
  }

  depends_on = [google_project_service.artifactregistry_api]
}

# -----------------------------------------------------------------------------
# Cloud Run Service - MNIST Classifier
# -----------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "mnist_classifier" {
  name     = var.app_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    # Cost optimization: scale to zero when not in use
    scaling {
      min_instance_count = var.min_instances  # 0 = scale to zero
      max_instance_count = var.max_instances  # Limit max to control costs
    }

    # Container configuration
    containers {
      image = local.container_image

      # Resource limits - keep minimal for cost savings
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle = true  # CPU only allocated during request processing
      }

      # Container port
      ports {
        container_port = 5000
      }

      # Environment variables
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      # Startup probe - wait for model to load (VLM downloads ~2GB, needs ~10 min)
      startup_probe {
        http_get {
          path = "/health"
          port = 5000
        }
        initial_delay_seconds = 120
        timeout_seconds       = 30
        period_seconds        = 60
        failure_threshold     = 10
      }

      # Liveness probe
      liveness_probe {
        http_get {
          path = "/health"
          port = 5000
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }

    # Timeout for cold starts (TensorFlow model loading takes time)
    timeout = "${var.request_timeout}s"

    # Service account
    service_account = google_service_account.cloudrun_sa.email

    labels = {
      environment = var.environment
      managed_by  = "terraform"
      lab         = "lab6"
    }
  }

  # Traffic configuration
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [
    google_project_service.run_api,
    google_artifact_registry_repository.mnist_repo
  ]
}

# -----------------------------------------------------------------------------
# IAM - Allow Public Access (unauthenticated)
# -----------------------------------------------------------------------------
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count = var.allow_public_access ? 1 : 0

  location = google_cloud_run_v2_service.mnist_classifier.location
  name     = google_cloud_run_v2_service.mnist_classifier.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# -----------------------------------------------------------------------------
# Service Account for Cloud Run
# -----------------------------------------------------------------------------
resource "google_service_account" "cloudrun_sa" {
  account_id   = "${var.app_name}-sa"
  display_name = "Cloud Run Service Account for MNIST Classifier"
  description  = "Minimal permissions service account for Cloud Run"
}

# Grant minimal permissions to service account
resource "google_project_iam_member" "cloudrun_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}
