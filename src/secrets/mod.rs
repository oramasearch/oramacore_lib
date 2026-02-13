pub mod aws;
mod cache;
pub mod local;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::{info, warn};

use aws::AwsSecretsConfig;
use cache::SecretsCache;
use local::LocalSecretsConfig;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SecretsProviderConfig {
    Aws(AwsSecretsConfig),
    Local(LocalSecretsConfig),
}

/// Each provider fetches raw key-value pairs and defines how to parse
/// its own key format into (collection_id, secret_key) pairs.
#[async_trait]
pub trait SecretsProvider: Send + Sync {
    async fn fetch_raw_secrets(&self) -> Result<HashMap<String, String>>;
    fn parse_key<'a>(&self, key: &'a str) -> Option<(&'a str, &'a str)>;
}

/// Manages multiple secrets providers, each with its own cache and TTL.
/// When looking up secrets for a collection, results from all providers are merged.
/// Later providers in the list override earlier ones on key conflicts.
pub struct SecretsService {
    caches: Vec<SecretsCache>,
}

impl SecretsService {
    pub fn empty() -> Arc<Self> {
        warn!("No secrets providers configured, secrets service will return empty results");
        Arc::new(Self { caches: vec![] })
    }

    pub async fn try_new(configs: Vec<SecretsProviderConfig>) -> Result<Arc<Self>> {
        let mut caches = Vec::with_capacity(configs.len());

        for config in configs {
            match config {
                SecretsProviderConfig::Aws(aws_config) => {
                    let ttl = aws_config.ttl;
                    let provider = aws::AwsSecretsProvider::try_new(&aws_config)
                        .await
                        .context("Failed to initialize AWS Secrets Manager provider")?;
                    let cache = SecretsCache::try_new(Box::new(provider), ttl)
                        .await
                        .context("Failed to initialize secrets cache for AWS provider")?;
                    caches.push(cache);
                }
                SecretsProviderConfig::Local(local_config) => {
                    let provider = local::LocalSecretsProvider::new(&local_config);
                    // Local providers never expire
                    let cache = SecretsCache::try_new(Box::new(provider), std::time::Duration::MAX)
                        .await
                        .context("Failed to initialize secrets cache for local provider")?;
                    caches.push(cache);
                }
            }
        }

        if caches.is_empty() {
            return Ok(Self::empty());
        }

        info!(
            provider_count = caches.len(),
            "Secrets service initialized with multiple providers"
        );

        Ok(Arc::new(Self { caches }))
    }

    /// Fetches secrets for a given collection by merging results from all providers.
    /// Later providers override earlier ones on key conflicts.
    pub async fn get_secrets_for_collection(
        &self,
        collection_id: &str,
    ) -> Arc<HashMap<String, String>> {
        if self.caches.is_empty() {
            return Arc::new(HashMap::new());
        }

        if self.caches.len() == 1 {
            // Fast path: single provider, no merging needed
            return self.caches[0].get_for_collection(collection_id).await;
        }

        // Merge results from all caches; later providers override earlier ones
        let mut merged = HashMap::new();
        for cache in &self.caches {
            let secrets = cache.get_for_collection(collection_id).await;
            for (key, value) in secrets.as_ref() {
                merged.insert(key.clone(), value.clone());
            }
        }

        Arc::new(merged)
    }
}
