use serde::{Deserialize, Serialize};

use crate::{pin_rules::PinRule};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub enum PinRuleOperation<DocumentId> {
    Insert(PinRule<DocumentId>),
    Delete(String),
}
