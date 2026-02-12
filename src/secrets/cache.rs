use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{error, info};

use super::SecretsProvider;

/// Internal cache data holding the fetched secrets and the timestamp of the last refresh.
struct CacheData {
    secrets: HashMap<String, String>,
    last_refresh: tokio::time::Instant,
}

/// In-memory cache for secrets fetched from external providers.
/// Lazily refreshes on access when the TTL expires.
pub struct SecretsCache {
    data: RwLock<CacheData>,
    providers: Vec<Box<dyn SecretsProvider>>,
    ttl: Duration,
}

impl SecretsCache {
    /// Creates a new SecretsCache with the given providers and TTL.
    /// Performs an initial fetch to populate the cache.
    pub async fn try_new(
        providers: Vec<Box<dyn SecretsProvider>>,
        ttl: Duration,
    ) -> anyhow::Result<Self> {
        let mut initial_secrets = HashMap::new();
        for provider in &providers {
            match provider.fetch_all_oramacore_secrets().await {
                Ok(secrets) => initial_secrets.extend(secrets),
                Err(e) => {
                    return Err(e).context("Failed initial secrets fetch");
                }
            }
        }

        info!(
            count = initial_secrets.len(),
            "Secrets cache initialized with initial fetch"
        );

        Ok(Self {
            data: RwLock::new(CacheData {
                secrets: initial_secrets,
                last_refresh: tokio::time::Instant::now(),
            }),
            providers,
            ttl,
        })
    }

    /// Refreshes the cache by fetching from all providers.
    /// On failure, logs the error and keeps stale data (graceful degradation).
    pub async fn refresh(&self) {
        let mut new_secrets = HashMap::new();
        let mut had_error = false;

        for provider in &self.providers {
            match provider.fetch_all_oramacore_secrets().await {
                Ok(secrets) => new_secrets.extend(secrets),
                Err(e) => {
                    error!(error = %e, "Failed to refresh secrets from provider, keeping stale data");
                    had_error = true;
                }
            }
        }

        // Only update if all providers succeeded, otherwise keep stale data
        if !had_error {
            let mut data = self.data.write().await;
            data.secrets = new_secrets;
            data.last_refresh = tokio::time::Instant::now();
            info!("Secrets cache refreshed successfully");
        }
    }

    /// Returns whether the cache TTL has expired.
    async fn is_expired(&self) -> bool {
        let data = self.data.read().await;
        data.last_refresh.elapsed() > self.ttl
    }

    /// Gets secrets for a specific collection by filtering on the `oramacore_{collection_id}_` prefix
    /// and stripping that prefix from the keys.
    /// Performs a lazy refresh if the cache is expired.
    pub async fn get_for_collection(&self, collection_id: &str) -> Arc<HashMap<String, String>> {
        if self.is_expired().await {
            self.refresh().await;
        }

        let prefix = format!("oramacore_{collection_id}_");
        let data = self.data.read().await;

        let filtered: HashMap<String, String> = data
            .secrets
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|stripped_key| (stripped_key.to_string(), value.clone()))
            })
            .collect();

        Arc::new(filtered)
    }
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Mock provider for testing
    struct MockProvider {
        secrets: HashMap<String, String>,
    }

    #[async_trait]
    impl SecretsProvider for MockProvider {
        async fn fetch_all_oramacore_secrets(&self) -> anyhow::Result<HashMap<String, String>> {
            Ok(self.secrets.clone())
        }
    }

    #[tokio::test]
    async fn test_cache_filters_by_collection_prefix() {
        let mut secrets = HashMap::new();
        secrets.insert("oramacore_col1_API_KEY".to_string(), "key123".to_string());
        secrets.insert(
            "oramacore_col1_DB_HOST".to_string(),
            "localhost".to_string(),
        );
        secrets.insert("oramacore_col2_API_KEY".to_string(), "key456".to_string());
        secrets.insert("unrelated_secret".to_string(), "value".to_string());

        let provider = MockProvider { secrets };
        let cache = SecretsCache::try_new(vec![Box::new(provider)], Duration::from_secs(300))
            .await
            .expect("Failed to create cache");

        // Get secrets for collection 1
        let col1_secrets = cache.get_for_collection("col1").await;
        assert_eq!(col1_secrets.len(), 2);
        assert_eq!(col1_secrets.get("API_KEY").unwrap(), "key123");
        assert_eq!(col1_secrets.get("DB_HOST").unwrap(), "localhost");

        // Get secrets for collection 2
        let col2_secrets = cache.get_for_collection("col2").await;
        assert_eq!(col2_secrets.len(), 1);
        assert_eq!(col2_secrets.get("API_KEY").unwrap(), "key456");

        // Get secrets for nonexistent collection
        let col3_secrets = cache.get_for_collection("col3").await;
        assert!(col3_secrets.is_empty());
    }

    #[tokio::test]
    async fn test_cache_empty_provider() {
        let provider = MockProvider {
            secrets: HashMap::new(),
        };
        let cache = SecretsCache::try_new(vec![Box::new(provider)], Duration::from_secs(300))
            .await
            .expect("Failed to create cache");

        let secrets = cache.get_for_collection("any").await;
        assert!(secrets.is_empty());
    }
}
