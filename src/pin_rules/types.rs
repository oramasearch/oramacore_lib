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
pub enum Condition {
    Is { pattern: String },
    StartsWith { pattern: String },
    Contains { pattern: String },
}

/// Required for the bincode deserialization
/// In fact, bincode doesn't support deserialize_any
/// So, we implemented it manually
#[derive(Serialize, Deserialize, Debug)]
struct SerdeCondition {
    anchoring: String,
    pattern: Option<String>,
}

impl Serialize for Condition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let c = match self {
            Condition::Is { pattern } => SerdeCondition {
                anchoring: "is".to_string(),
                pattern: Some(pattern.clone()),
            },
            Condition::StartsWith { pattern } => SerdeCondition {
                anchoring: "startsWith".to_string(),
                pattern: Some(pattern.clone()),
            },
            Condition::Contains { pattern } => SerdeCondition {
                anchoring: "contains".to_string(),
                pattern: Some(pattern.clone()),
            },
        };
        c.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Condition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let c = SerdeCondition::deserialize(deserializer)?;

        match c.anchoring.as_str() {
            "is" => {
                if let Some(pattern) = c.pattern {
                    Ok(Condition::Is { pattern })
                } else {
                    Err(serde::de::Error::custom("Unexpected pattern"))
                }
            }
            "startsWith" => {
                if let Some(pattern) = c.pattern {
                    Ok(Condition::StartsWith { pattern })
                } else {
                    Err(serde::de::Error::custom("Unexpected pattern"))
                }
            }
            "contains" => {
                if let Some(pattern) = c.pattern {
                    Ok(Condition::Contains { pattern })
                } else {
                    Err(serde::de::Error::custom("Unexpected pattern"))
                }
            }
            _ => Err(serde::de::Error::custom("Unexpected anchoring")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Anchoring {
    Is,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Consequence<DocId> {
    pub promote: Vec<PromoteItem<DocId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
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
            Condition::Is {
                pattern: "red jacket".to_string(),
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
                "conditions": [ { "pattern": "red", "anchoring": "startsWith" } ],
                "consequence": { "promote": [ { "doc_id": "RED_ITEM1", "position": 1 } ] }
            }"#;

        test_deserialization_logic(
            json,
            "promote-starts-with",
            Condition::StartsWith {
                pattern: "red".to_string(),
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
            Condition::Contains {
                pattern: "jacket".to_string(),
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
            Condition::Is {
                pattern: "exact match".to_string(),
            },
            Condition::StartsWith {
                pattern: "prefix".to_string(),
            },
            Condition::Contains {
                pattern: "middle".to_string(),
            },
        ];

        let serialized =
            serde_json::to_string_pretty(&conditions).expect("Failed to serialize conditions");

        let deserialized: Vec<Condition> =
            serde_json::from_str(&serialized).expect("Failed to deserialize conditions");
        assert_eq!(deserialized, conditions);
    }
}
