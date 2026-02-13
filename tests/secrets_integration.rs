use aws_credential_types::Credentials;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_secretsmanager::Client;
use aws_sdk_secretsmanager::config::{BehaviorVersion, Region};
use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};

use oramacore_lib::secrets::aws::AwsSecretsConfig;
use oramacore_lib::secrets::{SecretsManagerConfig, SecretsService};

async fn setup_localstack_secrets(
    seed_secrets: Vec<(&str, &str)>,
    ttl: std::time::Duration,
) -> (ContainerAsync<GenericImage>, Client, SecretsManagerConfig) {
    let localstack_port = 4566;

    let container = GenericImage::new("localstack/localstack", "latest")
        .with_exposed_port(ContainerPort::Tcp(localstack_port))
        .with_wait_for(WaitFor::message_on_stdout("Ready."))
        .with_env_var("SERVICES", "secretsmanager")
        .start()
        .await
        .expect("Failed to start LocalStack container");

    let host = container
        .get_host()
        .await
        .expect("Failed to get container host");
    let host_port = container
        .get_host_port_ipv4(ContainerPort::Tcp(localstack_port))
        .await
        .expect("Failed to get container port");
    let endpoint_url = format!("http://{host}:{host_port}");

    // Build a raw SDK client to seed test secrets
    let credentials = Credentials::new("test", "test", None, None, "localstack-test");
    let config = aws_config::SdkConfig::builder()
        .credentials_provider(SharedCredentialsProvider::new(credentials))
        .endpoint_url(&endpoint_url)
        .region(Region::new("us-east-1"))
        .behavior_version(BehaviorVersion::latest())
        .build();
    let client = Client::new(&config);

    // Seed the test secrets
    for (name, value) in &seed_secrets {
        client
            .create_secret()
            .name(*name)
            .secret_string(*value)
            .send()
            .await
            .unwrap_or_else(|e| panic!("Failed to seed secret '{name}': {e}"));
    }

    let secrets_config = SecretsManagerConfig {
        aws: Some(AwsSecretsConfig {
            region: "us-east-1".to_string(),
            ttl,
            access_key_id: Some("test".to_string()),
            secret_access_key: Some("test".to_string()),
            endpoint_url: Some(endpoint_url),
        }),
    };

    (container, client, secrets_config)
}

#[tokio::test]
async fn test_secrets_service_fetches_and_filters_by_collection() {
    let seed = vec![
        ("oramacore_col1_API_KEY", "key123"),
        ("oramacore_col1_DB_HOST", "db.example.com"),
        ("oramacore_col2_TOKEN", "tok456"),
        ("unrelated_secret", "nope"),
    ];

    let (_container, _client, config) =
        setup_localstack_secrets(seed, std::time::Duration::from_secs(300)).await;
    let service = SecretsService::try_new(config)
        .await
        .expect("Failed to create SecretsService");

    let col1 = service.get_secrets_for_collection("col1").await;
    assert_eq!(col1.len(), 2, "col1 should have 2 secrets, got: {col1:?}");
    assert_eq!(col1.get("API_KEY").expect("missing API_KEY"), "key123");
    assert_eq!(
        col1.get("DB_HOST").expect("missing DB_HOST"),
        "db.example.com"
    );

    let col2 = service.get_secrets_for_collection("col2").await;
    assert_eq!(col2.len(), 1, "col2 should have 1 secret, got: {col2:?}");
    assert_eq!(col2.get("TOKEN").expect("missing TOKEN"), "tok456");

    let col3 = service.get_secrets_for_collection("col3").await;
    assert!(col3.is_empty(), "col3 should be empty, got: {col3:?}");
}

#[tokio::test]
async fn test_secrets_service_empty_when_no_oramacore_secrets() {
    let seed = vec![
        ("some_other_secret", "value1"),
        ("another_secret", "value2"),
    ];

    let (_container, _client, config) =
        setup_localstack_secrets(seed, std::time::Duration::from_secs(300)).await;
    let service = SecretsService::try_new(config)
        .await
        .expect("Failed to create SecretsService");

    let result = service.get_secrets_for_collection("any").await;
    assert!(
        result.is_empty(),
        "Should return empty when no oramacore_ secrets exist, got: {result:?}"
    );
}

#[tokio::test]
async fn test_secrets_service_picks_up_updated_value_after_ttl() {
    let seed = vec![("oramacore_col1_API_KEY", "original_value")];
    let ttl = std::time::Duration::from_secs(1);

    let (_container, client, config) = setup_localstack_secrets(seed, ttl).await;
    let service = SecretsService::try_new(config)
        .await
        .expect("Failed to create SecretsService");

    let secrets = service.get_secrets_for_collection("col1").await;
    assert_eq!(
        secrets.get("API_KEY").expect("missing API_KEY"),
        "original_value"
    );

    client
        .put_secret_value()
        .secret_id("oramacore_col1_API_KEY")
        .secret_string("updated_value")
        .send()
        .await
        .expect("Failed to update secret");

    tokio::time::sleep(ttl + std::time::Duration::from_millis(500)).await;

    let secrets = service.get_secrets_for_collection("col1").await;
    assert_eq!(
        secrets.get("API_KEY").expect("missing API_KEY"),
        "updated_value",
        "Secret should reflect the updated value after TTL expiry"
    );
}

#[tokio::test]
async fn test_secrets_service_picks_up_deleted_secret_after_ttl() {
    let seed = vec![
        ("oramacore_col1_API_KEY", "key123"),
        ("oramacore_col1_DB_HOST", "db.example.com"),
    ];
    let ttl = std::time::Duration::from_secs(1);

    let (_container, client, config) = setup_localstack_secrets(seed, ttl).await;
    let service = SecretsService::try_new(config)
        .await
        .expect("Failed to create SecretsService");

    let secrets = service.get_secrets_for_collection("col1").await;
    assert_eq!(secrets.len(), 2, "col1 should have 2 secrets initially");

    client
        .delete_secret()
        .secret_id("oramacore_col1_API_KEY")
        .force_delete_without_recovery(true)
        .send()
        .await
        .expect("Failed to delete secret");

    tokio::time::sleep(ttl + std::time::Duration::from_millis(500)).await;

    let secrets = service.get_secrets_for_collection("col1").await;
    assert_eq!(
        secrets.len(),
        1,
        "col1 should have 1 secret after deletion, got: {secrets:?}"
    );
    assert!(
        secrets.get("API_KEY").is_none(),
        "API_KEY should be gone after deletion"
    );
    assert_eq!(
        secrets.get("DB_HOST").expect("missing DB_HOST"),
        "db.example.com",
        "DB_HOST should still exist"
    );
}
