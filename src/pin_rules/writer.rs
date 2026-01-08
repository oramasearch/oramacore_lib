use super::PinRule;
use super::file_util::get_rule_file_name;
use crate::fs::*;
use crate::pin_rules::file_util::{is_rule_file, remove_rule_file};
use anyhow::Context;
use debug_panic::debug_panic;
use std::{collections::HashMap, fmt::Debug, path::PathBuf};
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum PinRulesWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesWriter {
    rules: Vec<PinRule<String>>,
    rule_ids_to_delete: Vec<String>,
    has_uncommitted_changes: bool,
}

impl PinRulesWriter {
    pub fn empty() -> Result<Self, PinRulesWriterError> {
        Ok(Self {
            rules: Vec::new(),
            rule_ids_to_delete: Vec::new(),
            has_uncommitted_changes: false,
        })
    }

    pub fn try_new(data_dir: PathBuf) -> Result<Self, PinRulesWriterError> {
        create_if_not_exists(&data_dir)?;

        let dir = std::fs::read_dir(data_dir).context("Cannot read dir")?;

        let mut rules = Vec::new();
        for entry in dir {
            let Ok(entry) = entry else {
                debug_panic!("This shouldn't happen");
                error!("Error occurred while trying to read pin rule entry. Rule skipped");
                continue;
            };

            if is_rule_file(&entry.path()) {
                let rule = BufferedFile::open(entry.path())
                    .context("cannot open rules file")?
                    .read_json_data()
                    .context("cannot read rules file")?;
                rules.push(rule);
            }
        }

        Ok(Self {
            rules,
            rule_ids_to_delete: Vec::new(),
            has_uncommitted_changes: false,
        })
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), PinRulesWriterError> {
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

        self.has_uncommitted_changes = false;

        Ok(())
    }

    pub async fn insert_pin_rule(
        &mut self,
        rule: PinRule<String>,
    ) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != rule.id);
        self.rule_ids_to_delete.retain(|id| id != &rule.id);
        self.rules.push(rule);
        self.has_uncommitted_changes = true;

        Ok(())
    }

    pub async fn delete_pin_rule(&mut self, id: &str) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != id);
        self.rule_ids_to_delete.push(id.to_string());
        self.has_uncommitted_changes = true;

        Ok(())
    }

    pub fn list_pin_rules(&self) -> &[PinRule<String>] {
        &self.rules
    }

    pub fn get_involved_doc_ids(&self) -> Result<HashMap<String, String>, PinRulesWriterError> {
        let rules = self.list_pin_rules();

        let mut ret = HashMap::new();
        for rule in rules {
            for p in &rule.consequence.promote {
                ret.insert(p.doc_id.clone(), rule.id.clone());
            }
        }

        Ok(ret)
    }

    pub fn get_matching_rules(&self, doc_id_str: &str) -> Vec<PinRule<String>> {
        let rules = self.list_pin_rules();

        rules
            .iter()
            .filter(|rule| {
                rule.consequence
                    .promote
                    .iter()
                    .any(|p| p.doc_id == doc_id_str)
            })
            .cloned()
            .collect()
    }

    pub fn get_by_id(&self, rule_id: &str) -> Option<&PinRule<String>> {
        let rules = self.list_pin_rules();
        rules.iter().find(|r| r.id == rule_id)
    }

    pub fn has_pending_changes(&self) -> bool {
        self.has_uncommitted_changes
    }

    pub fn get_matching_rules_ids<'a, 's, 'd>(
        &'s self,
        doc_id_str: &'d str,
    ) -> impl Iterator<Item = String> + 'a
    where
        's: 'a,
        'd: 'a,
    {
        let rules = self.list_pin_rules();

        rules
            .iter()
            .filter(move |rule| {
                rule.consequence
                    .promote
                    .iter()
                    .any(|p| p.doc_id == doc_id_str)
            })
            .map(|rule| rule.id.clone())
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;
    use crate::fs::generate_new_path;
    use crate::pin_rules::Consequence;

    #[tokio::test]
    async fn test_simple() {
        let path = generate_new_path();

        let mut writer = PinRulesWriter::empty().unwrap();

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-1".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .unwrap();

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-2".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].id, "test-rule-1");
        assert_eq!(rules[1].id, "test-rule-2");

        writer.delete_pin_rule("test-rule-1").await.unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");

        writer.commit(path.clone()).unwrap();

        let writer = PinRulesWriter::try_new(path).unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");
    }

    #[tokio::test]
    async fn test_commit_pin_rules() {
        let base_dir = generate_new_path();
        let mut writer = PinRulesWriter::empty().unwrap();

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-1".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .expect("Failed to insert rule");

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-1");

        writer
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let mut writer =
            PinRulesWriter::try_new(base_dir.clone()).expect("Failed to create PinRulesWriter");

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-1");

        writer
            .delete_pin_rule("test-rule-1")
            .await
            .expect("Failed to delete rule");

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 0);

        writer
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let writer =
            PinRulesWriter::try_new(base_dir.clone()).expect("Failed to create PinRulesWriter");

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 0);
    }

    #[tokio::test]
    async fn test_has_pending_changes() {
        let path = generate_new_path();

        let mut writer = PinRulesWriter::empty().unwrap();

        assert!(!writer.has_pending_changes());

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-1".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .unwrap();
        assert!(writer.has_pending_changes());

        writer.commit(path.clone()).unwrap();
        assert!(!writer.has_pending_changes());

        writer.delete_pin_rule("test-rule-2").await.unwrap();
        assert!(writer.has_pending_changes());

        writer.commit(path.clone()).unwrap();
        assert!(!writer.has_pending_changes());
    }
}
