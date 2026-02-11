use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ValueOperation {
    Set { key: String, value: String },
    Delete { key: String },
}
