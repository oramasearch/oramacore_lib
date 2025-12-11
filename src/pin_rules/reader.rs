use std::{fmt::Debug, path::PathBuf};

use crate::fs::*;
use crate::nlp::TextParser;
use crate::pin_rules::file_util::{get_rule_file_name, is_rule_file, remove_rule_file};
use anyhow::Context;
use serde::{Serialize, de::DeserializeOwned};
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

fn get_token_stems_from_text(text: &str, text_parser: &TextParser) -> String {
    text_parser
        .tokenize_and_stem(text)
        .into_iter()
        .map(|(token, stems)| {
            if let Some(stem) = stems.first() {
                stem.to_owned()
            } else {
                token
            }
        })
        .collect()
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

    pub fn apply(&self, term: &str, text_parser: &TextParser) -> Vec<Consequence<DocumentId>> {
        let mut results = Vec::new();
        let mut term_stems_cache: Option<String> = None;

        for rule in &self.rules {
            for c in &rule.conditions {
                let matched = match c {
                    Condition::Is { pattern } => term == pattern,
                    Condition::StartsWith { pattern } => term.starts_with(pattern),
                    Condition::Contains { pattern } => term.contains(pattern),
                    Condition::IsStemmed { pattern } => {
                        let term_stems = term_stems_cache
                            .get_or_insert_with(|| get_token_stems_from_text(term, text_parser));
                        let pattern = get_token_stems_from_text(pattern, text_parser);

                        term_stems == &pattern
                    }
                    Condition::StartsWithStemmed { pattern } => {
                        let term_stems = term_stems_cache
                            .get_or_insert_with(|| get_token_stems_from_text(term, text_parser));
                        let pattern = get_token_stems_from_text(pattern, text_parser);

                        term_stems.starts_with(&pattern)
                    }
                    Condition::ContainsStemmed { pattern } => {
                        let term_stems = term_stems_cache
                            .get_or_insert_with(|| get_token_stems_from_text(term, text_parser));
                        let pattern = get_token_stems_from_text(pattern, text_parser);

                        term_stems.contains(&pattern)
                    }
                };

                if matched {
                    results.push(rule.consequence.clone());
                    break;
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use std::collections::HashSet;

    use super::*;
    use crate::nlp::locales::Locale;
    use crate::pin_rules::PromoteItem;

    #[test]
    fn test_pin_rules_reader_empty() {
        let reader: PinRulesReader<()> = PinRulesReader::empty();

        let ids = reader.get_rule_ids();
        assert_eq!(ids.len(), 0);

        let text_parser = TextParser::from_locale(Locale::EN);

        let consequences = reader.apply("test", &text_parser);
        assert!(consequences.is_empty());
    }

    #[test]
    fn test_commit_pin_rules() {
        let base_dir = generate_new_path();
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        let text_parser = TextParser::from_locale(Locale::EN);

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

        let consequences = reader.apply("term", &text_parser);
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test", &text_parser);
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let mut reader: PinRulesReader<usize> =
            PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("term", &text_parser);
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test", &text_parser);
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader
            .update(PinRuleOperation::Delete("test-rule-1".to_string()))
            .expect("Failed to delete rule");

        let consequences = reader.apply("test", &text_parser);
        assert_eq!(consequences.len(), 0);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit rules");

        let reader: PinRulesReader<usize> =
            PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("test", &text_parser);
        assert_eq!(consequences.len(), 0);
    }

    #[test]
    fn test_check_pin_rules() {
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        let text_parser = TextParser::from_locale(Locale::EN);

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-rules".to_string(),
                conditions: vec![
                    Condition::Is {
                        pattern: "test_is".to_string(),
                    },
                    Condition::StartsWith {
                        pattern: "test_start_with".to_string(),
                    },
                    Condition::Contains {
                        pattern: "test_contains".to_string(),
                    },
                ],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .expect("Failed to insert rule");

        let consequences = reader.apply("term", &text_parser);
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test_is", &text_parser);
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        let consequences = reader.apply("test_start_with_this_term", &text_parser);
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        let consequences = reader.apply("random_test_contains_text", &text_parser);
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);
    }

    #[test]
    fn test_check_pin_rules_stemmed() {
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        let text_parser = TextParser::from_locale(Locale::EN);

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-is-stemmed-rule".to_string(),
                conditions: vec![Condition::IsStemmed {
                    pattern: "shoes".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .expect("Failed to insert rule");

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-starts-with-stemmed-rule".to_string(),
                conditions: vec![Condition::StartsWithStemmed {
                    pattern: "shoes".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 2,
                        position: 2,
                    }],
                },
            }))
            .expect("Failed to insert rule");

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-contains-stemmed-rule".to_string(),
                conditions: vec![Condition::ContainsStemmed {
                    pattern: "shoes".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 3,
                        position: 3,
                    }],
                },
            }))
            .expect("Failed to insert rule");

        // user term: "shoe red", pin term: "shoes"
        // IsStemmed should not match because of "red"
        // StartsWithStemmed should match because "shoe" is a stem of "shoes"
        // ContainsStemmed should match because "shoe" is a stem of "shoes"
        let consequences = reader.apply("shoe red", &text_parser);
        assert_eq!(consequences.len(), 2);
        let promoted_docs: HashSet<usize> =
            consequences.iter().map(|c| c.promote[0].doc_id).collect();
        assert!(!promoted_docs.contains(&1));
        assert!(promoted_docs.contains(&2));
        assert!(promoted_docs.contains(&3));

        let consequences = reader.apply("shoe", &text_parser);
        assert_eq!(consequences.len(), 3);
        let promoted_docs: HashSet<usize> =
            consequences.iter().map(|c| c.promote[0].doc_id).collect();
        assert!(promoted_docs.contains(&1));
        assert!(promoted_docs.contains(&2));
        assert!(promoted_docs.contains(&3));

        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "test-is-stemmed-rule".to_string(),
                conditions: vec![Condition::IsStemmed {
                    pattern: "fruitless".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .expect("Failed to insert rule");
        let consequences = reader.apply("fruitlessly", &text_parser);
        assert_eq!(consequences.len(), 1);
    }

    #[test]
    fn test_symmetric_and_sentence_stemmed_match() {
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        let text_parser = TextParser::from_locale(Locale::EN);

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "is-stemmed-sentence".to_string(),
                conditions: vec![Condition::IsStemmed {
                    pattern: "a man walks".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .unwrap();

        let consequences = reader.apply("a man is walking", &text_parser);
        assert_eq!(consequences.len(), 1);

        let consequences = reader.apply("a man walked", &text_parser);
        assert_eq!(consequences.len(), 1);

        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "starts-with-stemmed-sentence".to_string(),
                conditions: vec![Condition::StartsWithStemmed {
                    pattern: "run shoe".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .unwrap();

        let consequences = reader.apply("running shoes for sale", &text_parser);
        assert_eq!(consequences.len(), 1);

        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();

        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "contains-stemmed-sentence".to_string(),
                conditions: vec![Condition::ContainsStemmed {
                    pattern: "run shoe".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .unwrap();

        let consequences = reader.apply("i like running shoes", &text_parser);
        assert_eq!(consequences.len(), 1);

        // Test symmetricity
        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "symmetric-is-stemmed".to_string(),
                conditions: vec![Condition::IsStemmed {
                    pattern: "shoes".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .unwrap();

        let consequences = reader.apply("shoe", &text_parser);
        assert_eq!(consequences.len(), 1);

        let mut reader: PinRulesReader<usize> = PinRulesReader::empty();
        reader
            .update(PinRuleOperation::Insert(PinRule {
                id: "symmetric-is-stemmed-2".to_string(),
                conditions: vec![Condition::IsStemmed {
                    pattern: "shoe".to_string(),
                }],
                consequence: Consequence {
                    promote: vec![PromoteItem {
                        doc_id: 1,
                        position: 1,
                    }],
                },
            }))
            .unwrap();

        let consequences = reader.apply("shoes", &text_parser);
        assert_eq!(consequences.len(), 1);
    }
}
