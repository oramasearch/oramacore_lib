use serde::{Deserialize, Serialize};

use super::{Shelf, ShelfId};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ShelfOperation<DocumentId> {
    Insert(Shelf<DocumentId>),
    Delete(ShelfId),
}
