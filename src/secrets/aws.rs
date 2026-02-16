use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_sdk_secretsmanager::Client;
use redact::Secret;
use serde::Deserialize;
use tracing::{info, warn};

use super::SecretsProvider;

#[derive(Debug, Clone, Deserialize)]
pub struct AwsSecretsConfig {
    pub region: String,
    #[serde(
        deserialize_with = "duration_str::deserialize_duration",
        default = "default_ttl"
    )]
    pub ttl: std::time::Duration,
    pub access_key_id: Option<Secret<String>>,
    pub secret_access_key: Option<Secret<String>>,
    pub endpoint_url: Option<String>,
}

fn default_ttl() -> std::time::Duration {
    std::time::Duration::from_secs(300)
}

pub const PROVIDER_NAME: &str = "oramacore-secrets";

/// Fetches secrets with the `oramacore_` prefix from AWS Secrets Manager.
/// Key format: `oramacore_{collection_id}_{secret_key}`.
pub struct AwsSecretsProvider {
    client: Client,
}

impl AwsSecretsProvider {
    pub async fn try_new(config: &AwsSecretsConfig) -> Result<Self> {
        let region = aws_sdk_secretsmanager::config::Region::new(config.region.clone());

        let mut aws_config_loader = aws_config::from_env().region(region);

        match (&config.access_key_id, &config.secret_access_key) {
            (Some(access_key_id), Some(secret_access_key)) => {
                let credentials = aws_sdk_secretsmanager::config::Credentials::new(
                    access_key_id.expose_secret(),
                    secret_access_key.expose_secret(),
                    None,
                    None,
                    PROVIDER_NAME,
                );
                aws_config_loader = aws_config_loader.credentials_provider(credentials);
                info!(region = %config.region, endpoint_url = ?config.endpoint_url, "AWS Secrets Manager provider initialized with explicit credentials");
            }
            _ => {
                info!(region = %config.region, endpoint_url = ?config.endpoint_url, "AWS Secrets Manager provider initialized with default credential chain (instance profile/env)");
            }
        }

        if let Some(endpoint_url) = &config.endpoint_url {
            aws_config_loader = aws_config_loader.endpoint_url(endpoint_url);
        }

        let aws_config = aws_config_loader.load().await;
        let client = Client::new(&aws_config);

        Ok(Self { client })
    }
}

#[async_trait]
impl SecretsProvider for AwsSecretsProvider {
    async fn fetch_raw_secrets(&self) -> Result<HashMap<String, String>> {
        let mut secrets = HashMap::new();

        let mut next_token: Option<String> = None;
        loop {
            let mut request = self.client.list_secrets().filters(
                aws_sdk_secretsmanager::types::Filter::builder()
                    .key(aws_sdk_secretsmanager::types::FilterNameStringType::Name)
                    .values("oramacore_")
                    .build(),
            );

            if let Some(token) = &next_token {
                request = request.next_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list secrets from AWS Secrets Manager")?;

            for secret_entry in response.secret_list() {
                let Some(secret_name) = secret_entry.name() else {
                    continue;
                };

                match self
                    .client
                    .get_secret_value()
                    .secret_id(secret_name)
                    .send()
                    .await
                {
                    Ok(value_response) => {
                        if let Some(secret_string) = value_response.secret_string() {
                            secrets.insert(secret_name.to_string(), secret_string.to_string());
                        }
                    }
                    Err(e) => {
                        warn!(
                            secret_name = %secret_name,
                            error = %e,
                            "Failed to fetch secret value, skipping"
                        );
                    }
                }
            }

            next_token = response.next_token().map(|s| s.to_string());
            if next_token.is_none() {
                break;
            }
        }

        info!(count = secrets.len(), "Fetched oramacore secrets from AWS");

        Ok(secrets)
    }

    fn parse_key<'a>(&self, key: &'a str) -> Option<(&'a str, &'a str)> {
        let rest = key.strip_prefix("oramacore_")?;
        let idx = rest.find('_')?;
        Some((&rest[..idx], &rest[idx + 1..]))
    }
}

#[cfg(test)]
mod tests {
    use redact::Secret;

    use super::AwsSecretsConfig;

    #[test]
    fn debug_does_not_print_aws_credentials() {
        let config = AwsSecretsConfig {
            region: "us-east-1".to_string(),
            ttl: std::time::Duration::from_secs(300),
            access_key_id: Some(Secret::new("AKIA_TEST_ACCESS_KEY".to_string())),
            secret_access_key: Some(Secret::new("TEST_SECRET_ACCESS_KEY_VALUE".to_string())),
            endpoint_url: Some("http://localhost:4566".to_string()),
        };

        let rendered = format!("{config:?}");

        println!("{rendered}");
        assert!(rendered.contains("AwsSecretsConfig"));
        assert!(rendered.contains("us-east-1"));
        assert!(rendered.contains("REDACTED"));
        assert!(!rendered.contains("AKIA_TEST_ACCESS_KEY"));
        assert!(!rendered.contains("TEST_SECRET_ACCESS_KEY_VALUE"));
    }
}
