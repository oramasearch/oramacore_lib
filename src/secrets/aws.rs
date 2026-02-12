use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_sdk_secretsmanager::Client;
use serde::Deserialize;
use tracing::{info, warn};

use super::SecretsProvider;

/// Configuration for AWS Secrets Manager provider.
/// Credentials default to the standard AWS credential chain (env vars, instance profile, etc.)
/// unless explicit access_key_id/secret_access_key are provided.
#[derive(Debug, Clone, Deserialize)]
pub struct AwsSecretsConfig {
    pub region: String,
    /// TTL for cache refresh. Parsed via duration-str (e.g. "5m", "30s").
    /// Defaults to 5 minutes if omitted.
    #[serde(
        deserialize_with = "duration_str::deserialize_duration",
        default = "default_ttl"
    )]
    pub ttl: std::time::Duration,
    /// Optional explicit AWS access key ID. If omitted, uses the default credential chain.
    pub access_key_id: Option<String>,
    /// Optional explicit AWS secret access key. If omitted, uses the default credential chain.
    pub secret_access_key: Option<String>,
}

fn default_ttl() -> std::time::Duration {
    std::time::Duration::from_secs(300) // 5 minutes
}

/// AWS Secrets Manager provider that fetches all secrets with the `oramacore_` prefix.
pub struct AwsSecretsProvider {
    client: Client,
}

impl AwsSecretsProvider {
    /// Creates a new AWS Secrets Manager provider.
    /// Uses explicit credentials if provided, otherwise falls back to the default AWS credential chain.
    pub async fn try_new(config: &AwsSecretsConfig) -> Result<Self> {
        let region = aws_sdk_secretsmanager::config::Region::new(config.region.clone());

        let aws_config = if let (Some(access_key_id), Some(secret_access_key)) =
            (&config.access_key_id, &config.secret_access_key)
        {
            // Use explicit credentials
            let credentials = aws_sdk_secretsmanager::config::Credentials::new(
                access_key_id,
                secret_access_key,
                None, // session token
                None, // expiry
                "oramacore-secrets",
            );
            aws_config::from_env()
                .region(region)
                .credentials_provider(credentials)
                .load()
                .await
        } else {
            // Use default credential chain (env vars, instance profile, etc.)
            aws_config::from_env().region(region).load().await
        };

        let client = Client::new(&aws_config);

        info!(region = %config.region, "AWS Secrets Manager provider initialized");

        Ok(Self { client })
    }
}

#[async_trait]
impl SecretsProvider for AwsSecretsProvider {
    /// Fetches all secrets whose names start with `oramacore_`.
    /// Uses list_secrets to find matching secrets, then get_secret_value for each.
    /// Individual fetch failures are logged as warnings but don't fail the entire operation.
    async fn fetch_all_oramacore_secrets(&self) -> Result<HashMap<String, String>> {
        let mut secrets = HashMap::new();

        // List all secrets with the oramacore_ prefix
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
                let secret_name = match secret_entry.name() {
                    Some(name) => name.to_string(),
                    None => continue,
                };

                // Fetch the actual secret value
                match self
                    .client
                    .get_secret_value()
                    .secret_id(&secret_name)
                    .send()
                    .await
                {
                    Ok(value_response) => {
                        if let Some(secret_string) = value_response.secret_string() {
                            secrets.insert(secret_name, secret_string.to_string());
                        }
                    }
                    Err(e) => {
                        // Log warning but continue fetching other secrets
                        warn!(
                            secret_name = %secret_name,
                            error = %e,
                            "Failed to fetch secret value, skipping"
                        );
                    }
                }
            }

            // Handle pagination
            next_token = response.next_token().map(|s| s.to_string());
            if next_token.is_none() {
                break;
            }
        }

        info!(count = secrets.len(), "Fetched oramacore secrets from AWS");

        Ok(secrets)
    }
}
