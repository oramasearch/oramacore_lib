use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use thiserror::Error;
use tracing::info;

use crate::fs::{BufferedFile, create_if_not_exists};

use super::{ValueOperation, ValuesDump};

#[derive(Error, Debug)]
pub enum ValuesReaderError {
    #[error("Io error {0:?}")]
    Io(std::io::Error),
    #[error("generic {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct ValuesReader {
    values: Arc<HashMap<String, String>>,
    has_uncommitted_changes: bool,
}

impl ValuesReader {
    /// Creates an empty values reader (for new collections).
    pub fn empty() -> Self {
        Self {
            values: Arc::new(HashMap::new()),
            has_uncommitted_changes: false,
        }
    }

    /// Loads values from disk.
    /// Returns an empty reader if the file does not exist.
    pub fn try_load(data_dir: PathBuf) -> Result<Self, ValuesReaderError> {
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

        info!("Loaded {} values from disk (reader)", values.len());

        Ok(Self {
            values: Arc::new(values),
            has_uncommitted_changes: false,
        })
    }

    /// Applies an operation immediately to the in-memory store.
    /// The change is visible right away but not persisted until commit().
    pub fn update(&mut self, op: ValueOperation) {
        match op {
            ValueOperation::Set { key, value } => {
                Arc::make_mut(&mut self.values).insert(key, value);
            }
            ValueOperation::Delete { key } => {
                Arc::make_mut(&mut self.values).remove(&key);
            }
        }
        self.has_uncommitted_changes = true;
    }

    /// Persists current values to disk.
    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), ValuesReaderError> {
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

    /// Gets a value by key.
    pub fn get(&self, key: &str) -> Option<String> {
        self.values.get(key).cloned()
    }

    /// Returns all values as a shared reference.
    pub fn list(&self) -> Arc<HashMap<String, String>> {
        Arc::clone(&self.values)
    }

    /// Returns the number of stored values.
    pub fn count(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::generate_new_path;

    #[test]
    fn test_empty_reader() {
        let reader = ValuesReader::empty();
        assert_eq!(reader.count(), 0);
        assert!(reader.list().is_empty());
        assert_eq!(reader.get("key"), None);
    }

    #[test]
    fn test_update_and_commit() {
        let base_dir = generate_new_path();
        let mut reader = ValuesReader::empty();

        reader.update(ValueOperation::Set {
            key: "key1".to_string(),
            value: "value1".to_string(),
        });
        reader.update(ValueOperation::Set {
            key: "key2".to_string(),
            value: "value2".to_string(),
        });

        // Values are applied immediately in-memory
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get("key1"), Some("value1".to_string()));
        assert_eq!(reader.get("key2"), Some("value2".to_string()));

        reader.commit(base_dir.clone()).expect("Failed to commit");

        // Reload from disk
        let reader = ValuesReader::try_load(base_dir).expect("Failed to load");
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get("key1"), Some("value1".to_string()));
    }

    #[test]
    fn test_delete_operation() {
        let base_dir = generate_new_path();
        let mut reader = ValuesReader::empty();

        reader.update(ValueOperation::Set {
            key: "key1".to_string(),
            value: "value1".to_string(),
        });
        reader.commit(base_dir.clone()).expect("Failed to commit");

        reader.update(ValueOperation::Delete {
            key: "key1".to_string(),
        });
        reader.commit(base_dir.clone()).expect("Failed to commit");

        assert_eq!(reader.count(), 0);
        assert_eq!(reader.get("key1"), None);

        // Reload from disk
        let reader = ValuesReader::try_load(base_dir).expect("Failed to load");
        assert_eq!(reader.count(), 0);
    }
}
