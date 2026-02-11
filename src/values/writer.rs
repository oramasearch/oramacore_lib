use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::fs::{BufferedFile, create_if_not_exists};

use super::{ValuesDump, MAX_KEYS, validate_key, validate_value};

#[derive(Error, Debug)]
pub enum ValuesWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct ValuesWriter {
    values: Arc<HashMap<String, String>>,
    has_uncommitted_changes: bool,
}

impl ValuesWriter {
    /// Creates an empty values writer (for new collections).
    pub fn empty() -> Self {
        Self {
            values: Arc::new(HashMap::new()),
            has_uncommitted_changes: false,
        }
    }

    /// Loads values from disk.
    /// Returns an empty writer if the file does not exist.
    pub fn try_load(data_dir: PathBuf) -> Result<Self, ValuesWriterError> {
        let file_path = data_dir.join("values.json");

        let values = if file_path.exists() {
            let dump: ValuesDump = BufferedFile::open(&file_path)
                .context("Cannot open values.json")?
                .read_json_data()
                .context("Cannot read values.json")?;

            dump.values
        } else {
            HashMap::new()
        };

        info!("Loaded {} values from disk", values.len());

        Ok(Self {
            values: Arc::new(values),
            has_uncommitted_changes: false,
        })
    }

    /// Sets a value. Validates key and value, checks max keys limit.
    /// Silently overwrites if the key already exists.
    pub fn set(&mut self, key: String, value: String) -> Result<(), ValuesWriterError> {
        debug!("Setting value: key='{}', value_size={}", key, value.len());

        validate_key(&key).context("Invalid value key")?;
        validate_value(&value).context("Invalid value")?;

        if self.values.len() >= MAX_KEYS && !self.values.contains_key(&key) {
            warn!("Cannot add key '{}': max keys ({}) reached", key, MAX_KEYS);
            return Err(anyhow::anyhow!(
                "Cannot add key '{}': maximum of {} keys allowed. Current count: {}",
                key,
                MAX_KEYS,
                self.values.len()
            )
            .into());
        }

        let is_update = self.values.contains_key(&key);
        Arc::make_mut(&mut self.values).insert(key.clone(), value);
        self.has_uncommitted_changes = true;

        info!(
            "Value {}: '{}'",
            if is_update { "updated" } else { "created" },
            key
        );

        Ok(())
    }

    /// Gets a value by key.
    pub fn get(&self, key: &str) -> Option<String> {
        self.values.get(key).cloned()
    }

    /// Deletes a value by key. Returns true if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        let removed = Arc::make_mut(&mut self.values).remove(key).is_some();
        if removed {
            self.has_uncommitted_changes = true;
            info!("Value deleted: '{}'", key);
        }
        removed
    }

    /// Returns all values as a shared reference.
    pub fn list(&self) -> Arc<HashMap<String, String>> {
        Arc::clone(&self.values)
    }

    /// Returns the number of stored values.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Whether there are uncommitted changes.
    pub fn has_pending_changes(&self) -> bool {
        self.has_uncommitted_changes
    }

    /// Persists values to disk. Only writes if there are uncommitted changes.
    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), ValuesWriterError> {
        if !self.has_uncommitted_changes {
            return Ok(());
        }

        create_if_not_exists(&data_dir)?;

        let dump = ValuesDump {
            values: (*self.values).clone(),
        };

        BufferedFile::create_or_overwrite(data_dir.join("values.json"))
            .context("Cannot create values.json")?
            .write_json_data(&dump)
            .context("Cannot write values.json")?;

        self.has_uncommitted_changes = false;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::generate_new_path;

    #[test]
    fn test_empty_writer() {
        let writer = ValuesWriter::empty();
        assert_eq!(writer.count(), 0);
        assert!(writer.list().is_empty());
        assert!(!writer.has_pending_changes());
    }

    #[test]
    fn test_set_and_get() {
        let mut writer = ValuesWriter::empty();

        writer
            .set("key1".to_string(), "value1".to_string())
            .unwrap();
        assert_eq!(writer.get("key1"), Some("value1".to_string()));
        assert_eq!(writer.count(), 1);
        assert!(writer.has_pending_changes());
    }

    #[test]
    fn test_overwrite() {
        let mut writer = ValuesWriter::empty();

        writer
            .set("key".to_string(), "original".to_string())
            .unwrap();
        writer
            .set("key".to_string(), "updated".to_string())
            .unwrap();
        assert_eq!(writer.get("key"), Some("updated".to_string()));
        assert_eq!(writer.count(), 1);
    }

    #[test]
    fn test_delete() {
        let mut writer = ValuesWriter::empty();

        writer.set("key".to_string(), "value".to_string()).unwrap();
        assert!(writer.delete("key"));
        assert_eq!(writer.get("key"), None);
        assert_eq!(writer.count(), 0);

        // Deleting non-existent key returns false
        assert!(!writer.delete("nonexistent"));
    }

    #[test]
    fn test_commit_and_reload() {
        let base_dir = generate_new_path();
        let mut writer = ValuesWriter::empty();

        writer
            .set("key1".to_string(), "value1".to_string())
            .unwrap();
        writer
            .set("key2".to_string(), "value2".to_string())
            .unwrap();

        writer.commit(base_dir.clone()).expect("Failed to commit");
        assert!(!writer.has_pending_changes());

        // Reload from disk
        let writer = ValuesWriter::try_load(base_dir).expect("Failed to load");
        assert_eq!(writer.count(), 2);
        assert_eq!(writer.get("key1"), Some("value1".to_string()));
        assert_eq!(writer.get("key2"), Some("value2".to_string()));
        assert!(!writer.has_pending_changes());
    }

    #[test]
    fn test_validation() {
        let mut writer = ValuesWriter::empty();

        // Empty key
        assert!(writer.set("".to_string(), "value".to_string()).is_err());
        // Key with spaces
        assert!(
            writer
                .set("bad key".to_string(), "value".to_string())
                .is_err()
        );
        // Empty value
        assert!(writer.set("key".to_string(), "".to_string()).is_err());
        // Value too large
        assert!(
            writer
                .set("key".to_string(), "x".repeat(10 * 1024 + 1))
                .is_err()
        );
    }

    #[test]
    fn test_has_pending_changes() {
        let path = generate_new_path();
        let mut writer = ValuesWriter::empty();

        assert!(!writer.has_pending_changes());

        writer.set("key".to_string(), "value".to_string()).unwrap();
        assert!(writer.has_pending_changes());

        writer.commit(path.clone()).unwrap();
        assert!(!writer.has_pending_changes());

        writer.delete("key");
        assert!(writer.has_pending_changes());

        writer.commit(path).unwrap();
        assert!(!writer.has_pending_changes());
    }
}
