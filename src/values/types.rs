use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MAX_KEYS: usize = 100;
pub const MIN_VALUE_SIZE: usize = 1;
pub const MAX_VALUE_SIZE: usize = 10 * 1024;
pub const MAX_KEY_LENGTH: usize = 128;

#[derive(Debug, Serialize, Deserialize)]
pub struct ValuesDump {
    pub values: HashMap<String, String>,
}

/// Validates a value key.
pub fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.len() > MAX_KEY_LENGTH {
        bail!("Key must be 1-{MAX_KEY_LENGTH} characters");
    }
    if !key
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        bail!("Key must contain only alphanumeric, underscore, or hyphen characters");
    }
    Ok(())
}

/// Validates a value.
pub fn validate_value(value: &str) -> Result<()> {
    let size = value.len();
    if !(MIN_VALUE_SIZE..=MAX_VALUE_SIZE).contains(&size) {
        bail!("Value must be {MIN_VALUE_SIZE}-{MAX_VALUE_SIZE} bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_key_valid() {
        assert!(validate_key("my-key").is_ok());
        assert!(validate_key("my_key").is_ok());
        assert!(validate_key("key123").is_ok());
        assert!(validate_key("a").is_ok());
        assert!(validate_key(&"k".repeat(128)).is_ok());
    }

    #[test]
    fn test_validate_key_invalid() {
        assert!(validate_key("").is_err());
        assert!(validate_key(&"k".repeat(129)).is_err());
        assert!(validate_key("key with spaces").is_err());
        assert!(validate_key("key@special").is_err());
        assert!(validate_key("key.dot").is_err());
    }

    #[test]
    fn test_validate_value_valid() {
        assert!(validate_value("a").is_ok());
        assert!(validate_value(&"x".repeat(10 * 1024)).is_ok());
    }

    #[test]
    fn test_validate_value_invalid() {
        assert!(validate_value("").is_err());
        assert!(validate_value(&"x".repeat(10 * 1024 + 1)).is_err());
    }
}
