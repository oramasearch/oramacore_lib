use std::{fmt::Debug, path::PathBuf};

use crate::pin_rules::file_util::{get_rule_file_name, is_rule_file, remove_rule_file};
use anyhow::Context;
use serde::{Serialize, de::DeserializeOwned};
use crate::fs::*;
use thiserror::Error;
use tracing::error;

use super::{Condition, Consequence, PinRule, PinRuleOperation};

#[derive(Error, Debug)]
pub enum PinRulesReaderError {
    #[error("Io error {0:?}")]
    Io(std::io::Error),
    #[error("generic {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesReader<DocumentId> {
    rules: Vec<PinRule<DocumentId>>,
    rule_ids_to_delete: Vec<String>,
}

impl<DocumentId: Serialize + DeserializeOwned + Debug + Clone> PinRulesReader<DocumentId> {
    pub fn empty() -> Self {
        Self {
            rules: Vec::new(),
            rule_ids_to_delete: Vec::new(),
        }
    }

    pub fn try_new(data_dir: PathBuf) -> Result<Self, PinRulesReaderError> {
        create_if_not_exists(&data_dir)?;

        let rules: Vec<_> = std::fs::read_dir(&data_dir)
            .context("Cannot read pin rules directory")?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if is_rule_file(&path) {
                    let rule: PinRule<DocumentId> =
                        BufferedFile::open(path).ok()?.read_json_data().ok()?;
                    Some(rule)
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            rules,
            rule_ids_to_delete: Vec::new(),
        })
    }

    pub fn update(&mut self, op: PinRuleOperation<DocumentId>) -> Result<(), PinRulesReaderError> {
        match op {
            PinRuleOperation::Insert(rule) => {
                self.rule_ids_to_delete.retain(|id| id != &rule.id);
                self.rules.retain(|r| r.id != rule.id);
                self.rules.push(rule);
            }
            PinRuleOperation::Delete(rule_id) => {
                self.rules.retain(|r| r.id != rule_id);
                self.rule_ids_to_delete.push(rule_id);
            }
        }
        Ok(())
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), PinRulesReaderError> {
        create_if_not_exists(&data_dir)?;

        for rule in &self.rules {
            let file_path = data_dir.join(get_rule_file_name(&rule.id));
            BufferedFile::create_or_overwrite(file_path)
                .context("Cannot create file")?
                .write_json_data(rule)
                .context("Cannot write rule to file")?;
        }

        for rule_id_to_remove in self.rule_ids_to_delete.drain(..) {
            let file_path = data_dir.join(get_rule_file_name(&rule_id_to_remove));
            remove_rule_file(file_path);
        }

        Ok(())
    }

    pub fn get_rule_ids(&self) -> Vec<String> {
        self.rules.iter().map(|r| r.id.clone()).collect()
    }

    pub fn apply(&self, term: &str) -> Vec<Consequence<DocumentId>> {
        let mut results = Vec::new();
        for rule in &self.rules {
            for c in &rule.conditions {
                match c {
                    Condition::Is { pattern } if pattern == term => {
                        results.push(rule.consequence.clone());
                        break;
                    }
                    _ => continue,
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;
    use crate::pin_rules::PromoteItem;

    #[test]
    fn test_pin_rules_reader_empty() {
        let reader: PinRulesReader<()> = PinRulesReader::empty();

        let ids = reader.get_rule_ids();
        assert_eq!(ids.len(), 0);

        // Test applying rules
        let consequences = reader.apply("test");
        assert!(consequences.is_empty());
    }

    #[test]
    fn test_apply_pin_rules() {
        let base_dir = generate_new_path();
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-rule-1".to_string(),
                conditions: vec![Condition::Is {
                    pattern: "test".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .expect("Failed to insert rule");

        let consequences = reader.apply("term");
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test");
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let mut reader: PinRulesReader<usize> =
            PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("term");
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test");
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader
            .update(PinRuleOperation::Delete("test-rule-1".to_string()))
            .expect("Failed to delete rule");

        let consequences = reader.apply("test");
        assert_eq!(consequences.len(), 0);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let reader: PinRulesReader<usize> =
            PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("test");
        assert_eq!(consequences.len(), 0);
    }
}
