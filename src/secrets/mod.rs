pub mod aws;
pub mod cache;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use aws::AwsSecretsConfig;
use cache::SecretsCache;

/// Configuration for the secrets manager.
#[derive(Debug, Clone, Deserialize)]
pub struct SecretsManagerConfig {
    pub aws: Option<AwsSecretsConfig>,
}

/// Trait for fetching secrets from an external provider.
/// Each provider implementation fetches all secrets with the `oramacore_` prefix.
#[async_trait]
pub trait SecretsProvider: Send + Sync {
    async fn fetch_all_oramacore_secrets(&self) -> Result<HashMap<String, String>>;
}

/// Service that manages secrets from multiple providers with caching.
pub struct SecretsService {
    cache: Arc<SecretsCache>,
}

impl SecretsService {
    /// Creates a new SecretsService from configuration.
    /// Initializes all configured providers and performs an initial fetch.
    /// Secrets are lazily refreshed on access when the TTL expires.
    pub async fn try_new(config: SecretsManagerConfig) -> Result<Arc<Self>> {
        let mut providers: Vec<Box<dyn SecretsProvider>> = Vec::new();
        let mut ttl = std::time::Duration::from_secs(300);

        // Initialize AWS provider if configured
        if let Some(aws_config) = &config.aws {
            ttl = aws_config.ttl;
            let aws_provider = aws::AwsSecretsProvider::try_new(aws_config)
                .await
                .context("Failed to initialize AWS Secrets Manager provider")?;
            providers.push(Box::new(aws_provider));
        }

        if providers.is_empty() {
            anyhow::bail!(
                "No secrets providers configured. At least one provider (e.g., aws) must be specified."
            );
        }

        let cache = SecretsCache::try_new(providers, ttl)
            .await
            .context("Failed to initialize secrets cache")?;
        let cache = Arc::new(cache);

        info!("Secrets service initialized");

        Ok(Arc::new(Self { cache }))
    }

    /// Gets secrets for a specific collection.
    /// Returns a filtered and prefix-stripped HashMap wrapped in Arc.
    pub async fn get_secrets_for_collection(
        &self,
        collection_id: &str,
    ) -> Arc<HashMap<String, String>> {
        self.cache.get_for_collection(collection_id).await
    }
}

/// Returns an empty secrets HashMap wrapped in Arc.
/// Used when no secrets manager is configured.
pub fn empty_secrets() -> Arc<HashMap<String, String>> {
    Arc::new(HashMap::new())
}
