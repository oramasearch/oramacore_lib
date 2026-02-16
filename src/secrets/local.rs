use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use super::SecretsProvider;

#[derive(Debug, Clone, Deserialize)]
pub struct LocalSecretsConfig {
    /// Key format: `{collection_id}_{secret_key}`.
    pub secrets: HashMap<String, String>,
}

/// In-memory secrets provider. Static set of secrets, never changes on refresh.
pub struct LocalSecretsProvider {
    secrets: HashMap<String, String>,
}

impl LocalSecretsProvider {
    pub fn new(config: &LocalSecretsConfig) -> Self {
        info!(
            count = config.secrets.len(),
            "Local secrets provider initialized"
        );
        Self {
            secrets: config.secrets.clone(),
        }
    }
}

#[async_trait]
impl SecretsProvider for LocalSecretsProvider {
    async fn fetch_raw_secrets(&self) -> Result<HashMap<String, String>> {
        Ok(self.secrets.clone())
    }

    fn parse_key<'a>(&self, key: &'a str) -> Option<(&'a str, &'a str)> {
        let idx = key.find('_')?;
        Some((&key[..idx], &key[idx + 1..]))
    }
}
