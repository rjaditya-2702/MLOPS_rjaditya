# =============================================================================
# Terraform GCP Lab - Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# Service URLs
# -----------------------------------------------------------------------------

output "service_url" {
  description = "URL of the deployed MNIST classifier"
  value       = google_cloud_run_v2_service.mnist_classifier.uri
}

output "health_check_url" {
  description = "Health check endpoint"
  value       = "${google_cloud_run_v2_service.mnist_classifier.uri}/health"
}

# -----------------------------------------------------------------------------
# Artifact Registry
# -----------------------------------------------------------------------------

output "artifact_registry_repo" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mnist_repo.repository_id}"
}

output "docker_push_command" {
  description = "Command to push Docker image"
  value       = "docker push ${local.container_image}"
}

output "container_image" {
  description = "Full container image URL"
  value       = local.container_image
}

# -----------------------------------------------------------------------------
# Service Details
# -----------------------------------------------------------------------------

output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.mnist_classifier.name
}

output "service_location" {
  description = "Cloud Run service location"
  value       = google_cloud_run_v2_service.mnist_classifier.location
}

output "latest_revision" {
  description = "Latest revision name"
  value       = google_cloud_run_v2_service.mnist_classifier.latest_ready_revision
}

# -----------------------------------------------------------------------------
# Configuration Summary
# -----------------------------------------------------------------------------

output "resource_config" {
  description = "Resource configuration"
  value = {
    cpu           = var.cpu_limit
    memory        = var.memory_limit
    min_instances = var.min_instances
    max_instances = var.max_instances
  }
}

output "cost_info" {
  description = "Cost optimization info"
  value       = var.min_instances == 0 ? "✅ Scale to zero enabled - NO COST when idle!" : "⚠️ Min instances > 0 - will incur charges even when idle"
}

# -----------------------------------------------------------------------------
# Helpful Commands
# -----------------------------------------------------------------------------

output "test_commands" {
  description = "Commands to test the deployment"
  value = <<-EOT

    # Test health endpoint
    curl ${google_cloud_run_v2_service.mnist_classifier.uri}/health

    # Open in browser
    open ${google_cloud_run_v2_service.mnist_classifier.uri}

    # View logs
    gcloud run services logs read ${var.app_name} --region=${var.region} --limit=50

    # Describe service
    gcloud run services describe ${var.app_name} --region=${var.region}

  EOT
}

# -----------------------------------------------------------------------------
# Deployment Summary
# -----------------------------------------------------------------------------

output "deployment_summary" {
  description = "Deployment summary"
  value = <<-EOT

    ╔══════════════════════════════════════════════════════════════════════╗
    ║           🚀 MNIST CLASSIFIER DEPLOYED TO CLOUD RUN                  ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  Service: ${google_cloud_run_v2_service.mnist_classifier.name}
    ║  Region:  ${var.region}
    ║  URL:     ${google_cloud_run_v2_service.mnist_classifier.uri}
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  Resources: ${var.cpu_limit} CPU, ${var.memory_limit} Memory
    ║  Scaling:   ${var.min_instances} - ${var.max_instances} instances
    ║  ${var.min_instances == 0 ? "💰 Cost: Scales to zero - FREE when idle!" : "⚠️  Minimum instances always running"}
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝

    📝 First request may be slow (cold start while loading TensorFlow model)

  EOT
}
