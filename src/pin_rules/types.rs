use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::future::Future;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct PinRule<DocId> {
    pub id: String,
    pub conditions: Vec<Condition>,
    pub consequence: Consequence<DocId>,
}

impl<DocId> PinRule<DocId> {
    pub async fn convert_ids<F, NewId>(self, f: F) -> PinRule<NewId>
    where
        F: Fn(DocId) -> NewId,
    {
        PinRule {
            id: self.id,
            conditions: self.conditions,
            consequence: Consequence {
                promote: self
                    .consequence
                    .promote
                    .into_iter()
                    .map(|item| {
                        let new_doc_id = f(item.doc_id);
                        PromoteItem {
                            doc_id: new_doc_id,
                            position: item.position,
                        }
                    })
                    .collect(),
            },
        }
    }

    pub async fn async_convert_ids<F, Fut, NewId>(self, f: F) -> PinRule<NewId>
    where
        F: Fn(DocId) -> Fut + Send,
        Fut: Future<Output = NewId> + Send,
    {
        use futures::StreamExt;

        PinRule {
            id: self.id,
            conditions: self.conditions,
            consequence: Consequence {
                promote: futures::stream::iter(self.consequence.promote)
                    .then(|item| {
                        let doc_id_fut = f(item.doc_id);
                        async move {
                            PromoteItem {
                                doc_id: doc_id_fut.await,
                                position: item.position,
                            }
                        }
                    })
                    .collect()
                    .await,
            },
        }
    }
}

impl TryFrom<serde_json::Value> for PinRule<String> {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> anyhow::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Anchoring {
    Is,
    StartsWith,
    Contains,
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Normalization {
    None,
    Stem,
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Condition {
    pub pattern: String,
    pub anchoring: Anchoring,
    pub normalization: Normalization,
}

/// Required for the bincode deserialization
/// In fact, bincode doesn't support deserialize_any
/// So, we implemented it manually
#[derive(Serialize, Deserialize, Debug)]
struct SerdeCondition {
    anchoring: String,
    pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    normalization: Option<String>,
}

impl Serialize for Condition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let anchoring = match self.anchoring {
            Anchoring::Is => "is",
            Anchoring::StartsWith => "startsWith",
            Anchoring::Contains => "contains",
        };

        let normalization = match self.normalization {
            Normalization::None => None,
            Normalization::Stem => Some("stem".to_string()),
        };

        SerdeCondition {
            pattern: self.pattern.clone(),
            anchoring: anchoring.to_string(),
            normalization,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Condition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let c = SerdeCondition::deserialize(deserializer)?;

        let anchoring = match c.anchoring.as_str() {
            "is" => Anchoring::Is,
            "startsWith" => Anchoring::StartsWith,
            "contains" => Anchoring::Contains,
            anchor => {
                return Err(serde::de::Error::custom(format!(
                    "Unexpected anchoring '{anchor}'"
                )));
            }
        };

        let normalization = match c.normalization {
            None => Normalization::None,
            Some(t) => match t.as_str() {
                "stem" => Normalization::Stem,
                norm => {
                    return Err(serde::de::Error::custom(format!(
                        "Unexpected normalization '{norm}'"
                    )));
                }
            },
        };

        Ok(Condition {
            pattern: c.pattern,
            anchoring,
            normalization,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Consequence<DocId> {
    pub promote: Vec<PromoteItem<DocId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromoteItem<DocId> {
    pub doc_id: DocId,
    pub position: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_deserialization_logic(
        json: &str,
        expected_id: &str,
        expected_condition: Condition,
        expected_promote: Vec<PromoteItem<String>>,
    ) {
        let pin_rule: PinRule<String> =
            serde_json::from_str(json).expect("Failed to deserialize JSON");
        assert_eq!(pin_rule.id, expected_id);
        assert_eq!(pin_rule.conditions.len(), 1);
        assert_eq!(pin_rule.conditions[0], expected_condition);
        assert_eq!(pin_rule.consequence.promote, expected_promote);
    }

    #[test]
    fn test_is_deserialization() {
        let json = r#"{
                "id": "promote-red-jacket",
                "conditions": [ { "pattern": "red jacket", "anchoring": "is" } ],
                "consequence": {
                    "promote": [
                        { "doc_id": "JACKET42", "position": 1 },
                        { "doc_id": "PANTS77", "position": 2 }
                    ]
                }
            }"#;

        test_deserialization_logic(
            json,
            "promote-red-jacket",
            Condition {
                pattern: "red jacket".to_string(),
                anchoring: Anchoring::Is,
                normalization: Normalization::None,
            },
            vec![
                PromoteItem {
                    doc_id: "JACKET42".to_string(),
                    position: 1,
                },
                PromoteItem {
                    doc_id: "PANTS77".to_string(),
                    position: 2,
                },
            ],
        );
    }

    #[test]
    fn test_startswith_deserialization() {
        let json = r#"{
                "id": "promote-starts-with",
                "conditions": [ { "pattern": "red", "anchoring": "startsWith", "normalization": "stem" } ],
                "consequence": { "promote": [ { "doc_id": "RED_ITEM1", "position": 1 } ] }
            }"#;

        test_deserialization_logic(
            json,
            "promote-starts-with",
            Condition {
                pattern: "red".to_string(),
                anchoring: Anchoring::StartsWith,
                normalization: Normalization::Stem,
            },
            vec![PromoteItem {
                doc_id: "RED_ITEM1".to_string(),
                position: 1,
            }],
        );
    }

    #[test]
    fn test_contains_deserialization() {
        let json = r#"{
                "id": "promote-contains",
                "conditions": [ { "pattern": "jacket", "anchoring": "contains" } ],
                "consequence": {
                    "promote": [
                        { "doc_id": "JACKET_ITEM1", "position": 1 },
                        { "doc_id": "JACKET_ITEM2", "position": 2 }
                    ]
                }
            }"#;

        test_deserialization_logic(
            json,
            "promote-contains",
            Condition {
                pattern: "jacket".to_string(),
                anchoring: Anchoring::Contains,
                normalization: Normalization::None,
            },
            vec![
                PromoteItem {
                    doc_id: "JACKET_ITEM1".to_string(),
                    position: 1,
                },
                PromoteItem {
                    doc_id: "JACKET_ITEM2".to_string(),
                    position: 2,
                },
            ],
        );
    }

    #[test]
    fn test_condition_serialization_roundtrip() {
        let conditions = vec![
            Condition {
                pattern: "exact match".to_string(),
                anchoring: Anchoring::Is,
                normalization: Normalization::None,
            },
            Condition {
                pattern: "prefix".to_string(),
                anchoring: Anchoring::StartsWith,
                normalization: Normalization::None,
            },
            Condition {
                pattern: "middle".to_string(),
                anchoring: Anchoring::Contains,
                normalization: Normalization::None,
            },
            Condition {
                pattern: "exact match stemmed".to_string(),
                anchoring: Anchoring::Is,
                normalization: Normalization::Stem,
            },
            Condition {
                pattern: "prefix stemmed".to_string(),
                anchoring: Anchoring::StartsWith,
                normalization: Normalization::Stem,
            },
            Condition {
                pattern: "middle stemmed".to_string(),
                anchoring: Anchoring::Contains,
                normalization: Normalization::Stem,
            },
        ];

        let serialized =
            serde_json::to_string_pretty(&conditions).expect("Failed to serialize conditions");

        let deserialized: Vec<Condition> =
            serde_json::from_str(&serialized).expect("Failed to deserialize conditions");
        assert_eq!(deserialized, conditions);
    }
}
