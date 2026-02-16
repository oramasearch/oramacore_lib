use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use super::SecretsProvider;

#[derive(Clone, Deserialize)]
pub struct LocalSecretsConfig {
    /// Key format: `{collection_id}_{secret_key}`.
    pub secrets: HashMap<String, String>,
}

impl std::fmt::Debug for LocalSecretsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSecretsConfig")
            .field("secrets", &format!("[{} entries]", self.secrets.len()))
            .finish()
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::LocalSecretsConfig;

    #[test]
    fn debug_does_not_print_secret_values() {
        let mut secrets = HashMap::new();
        secrets.insert("col1_api_key".to_string(), "super-secret-value".to_string());
        secrets.insert("col1_token".to_string(), "another-secret-value".to_string());

        let config = LocalSecretsConfig { secrets };
        let rendered = format!("{config:?}");

        assert!(rendered.contains("LocalSecretsConfig"));
        assert!(rendered.contains("[2 entries]"));
        assert!(!rendered.contains("super-secret-value"));
        assert!(!rendered.contains("another-secret-value"));
    }
}
