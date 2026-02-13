pub mod aws;
mod cache;
pub mod local;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use aws::AwsSecretsConfig;
use cache::SecretsCache;
use local::LocalSecretsConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct SecretsManagerConfig {
    pub aws: Option<AwsSecretsConfig>,
    pub local: Option<LocalSecretsConfig>,
}

/// Each provider fetches raw key-value pairs and defines how to parse
/// its own key format into (collection_id, secret_key) pairs.
#[async_trait]
pub trait SecretsProvider: Send + Sync {
    async fn fetch_raw_secrets(&self) -> Result<HashMap<String, String>>;
    fn parse_key<'a>(&self, key: &'a str) -> Option<(&'a str, &'a str)>;
}

pub struct SecretsService {
    cache: SecretsCache,
}

impl SecretsService {
    pub async fn try_new(config: SecretsManagerConfig) -> Result<Arc<Self>> {
        let mut providers: Vec<Box<dyn SecretsProvider>> = Vec::new();
        let mut ttl = std::time::Duration::from_secs(300);

        if let Some(aws_config) = &config.aws {
            ttl = aws_config.ttl;
            let aws_provider = aws::AwsSecretsProvider::try_new(aws_config)
                .await
                .context("Failed to initialize AWS Secrets Manager provider")?;
            providers.push(Box::new(aws_provider));
        }

        if let Some(local_config) = &config.local {
            let local_provider = local::LocalSecretsProvider::new(local_config);
            providers.push(Box::new(local_provider));
        }

        if providers.is_empty() {
            anyhow::bail!(
                "No secrets providers configured. At least one provider (e.g., aws, local) must be specified."
            );
        }

        let cache = SecretsCache::try_new(providers, ttl)
            .await
            .context("Failed to initialize secrets cache")?;

        info!("Secrets service initialized");

        Ok(Arc::new(Self { cache }))
    }

    pub async fn get_secrets_for_collection(
        &self,
        collection_id: &str,
    ) -> Arc<HashMap<String, String>> {
        self.cache.get_for_collection(collection_id).await
    }
}
