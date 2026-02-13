use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::sync::RwLock;
use tracing::{error, info};

use super::SecretsProvider;

struct CacheData {
    per_collection: HashMap<String, Arc<HashMap<String, String>>>,
    last_refresh: tokio::time::Instant,
}

fn group_raw_secrets(
    provider: &dyn SecretsProvider,
    raw: &HashMap<String, String>,
    grouped: &mut HashMap<String, HashMap<String, String>>,
) {
    for (key, value) in raw {
        if let Some((collection_id, secret_key)) = provider.parse_key(key) {
            if let Some(map) = grouped.get_mut(collection_id) {
                map.insert(secret_key.to_string(), value.clone());
            } else {
                let mut map = HashMap::new();
                map.insert(secret_key.to_string(), value.clone());
                grouped.insert(collection_id.to_string(), map);
            }
        }
    }
}

fn wrap_in_arc(
    grouped: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, Arc<HashMap<String, String>>> {
    grouped.into_iter().map(|(k, v)| (k, Arc::new(v))).collect()
}

pub(super) struct SecretsCache {
    data: RwLock<CacheData>,
    providers: Vec<Box<dyn SecretsProvider>>,
    ttl: Duration,
}

impl SecretsCache {
    pub(super) async fn try_new(
        providers: Vec<Box<dyn SecretsProvider>>,
        ttl: Duration,
    ) -> anyhow::Result<Self> {
        let mut grouped: HashMap<String, HashMap<String, String>> = HashMap::new();
        for provider in &providers {
            let raw = provider
                .fetch_raw_secrets()
                .await
                .context("Failed initial secrets fetch")?;
            group_raw_secrets(provider.as_ref(), &raw, &mut grouped);
        }

        let total: usize = grouped.values().map(|m| m.len()).sum();
        info!(
            count = total,
            "Secrets cache initialized with initial fetch"
        );

        let per_collection = wrap_in_arc(grouped);

        Ok(Self {
            data: RwLock::new(CacheData {
                per_collection,
                last_refresh: tokio::time::Instant::now(),
            }),
            providers,
            ttl,
        })
    }

    async fn refresh(&self) {
        let mut grouped: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut had_error = false;

        for provider in &self.providers {
            match provider.fetch_raw_secrets().await {
                Ok(raw) => {
                    group_raw_secrets(provider.as_ref(), &raw, &mut grouped);
                }
                Err(e) => {
                    error!(error = %e, "Failed to refresh secrets from provider, keeping stale data");
                    had_error = true;
                }
            }
        }

        if !had_error {
            let per_collection = wrap_in_arc(grouped);
            let mut data = self.data.write().await;
            data.per_collection = per_collection;
            data.last_refresh = tokio::time::Instant::now();
            info!("Secrets cache refreshed successfully");
        }
    }

    async fn is_expired(&self) -> bool {
        let data = self.data.read().await;
        data.last_refresh.elapsed() > self.ttl
    }

    pub(super) async fn get_for_collection(
        &self,
        collection_id: &str,
    ) -> Arc<HashMap<String, String>> {
        if self.is_expired().await {
            self.refresh().await;
        }

        let data = self.data.read().await;
        data.per_collection
            .get(collection_id)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct MockProvider {
        secrets: HashMap<String, String>,
        prefix: &'static str,
    }

    #[async_trait]
    impl SecretsProvider for MockProvider {
        async fn fetch_raw_secrets(&self) -> anyhow::Result<HashMap<String, String>> {
            Ok(self.secrets.clone())
        }

        fn parse_key<'a>(&self, key: &'a str) -> Option<(&'a str, &'a str)> {
            let rest = key.strip_prefix(self.prefix)?.strip_prefix('_')?;
            let idx = rest.find('_')?;
            Some((&rest[..idx], &rest[idx + 1..]))
        }
    }

    #[tokio::test]
    async fn test_cache_filters_by_collection() {
        let mut secrets = HashMap::new();
        secrets.insert("test_col1_API_KEY".to_string(), "key123".to_string());
        secrets.insert("test_col1_DB_HOST".to_string(), "localhost".to_string());
        secrets.insert("test_col2_API_KEY".to_string(), "key456".to_string());
        secrets.insert("unrelated_secret".to_string(), "value".to_string());

        let provider = MockProvider {
            secrets,
            prefix: "test",
        };
        let cache = SecretsCache::try_new(vec![Box::new(provider)], Duration::from_secs(300))
            .await
            .expect("Failed to create cache");

        let col1_secrets = cache.get_for_collection("col1").await;
        assert_eq!(col1_secrets.len(), 2);
        assert_eq!(col1_secrets.get("API_KEY").unwrap(), "key123");
        assert_eq!(col1_secrets.get("DB_HOST").unwrap(), "localhost");

        let col2_secrets = cache.get_for_collection("col2").await;
        assert_eq!(col2_secrets.len(), 1);
        assert_eq!(col2_secrets.get("API_KEY").unwrap(), "key456");

        let col3_secrets = cache.get_for_collection("col3").await;
        assert!(col3_secrets.is_empty());
    }

    #[tokio::test]
    async fn test_cache_empty_provider() {
        let provider = MockProvider {
            secrets: HashMap::new(),
            prefix: "test",
        };
        let cache = SecretsCache::try_new(vec![Box::new(provider)], Duration::from_secs(300))
            .await
            .expect("Failed to create cache");

        let secrets = cache.get_for_collection("any").await;
        assert!(secrets.is_empty());
    }
}
