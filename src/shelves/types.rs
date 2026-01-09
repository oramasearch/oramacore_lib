use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShelfIdError {
    #[error("Shelf id must be at least 2 characters")]
    TooShort,
    #[error("Shelf id must be at most 64 characters")]
    TooLong,
    #[error("Shelf id must contain only alphanumeric characters, dashes or underscores")]
    InvalidCharacters,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShelfId(String);
impl ShelfId {
    pub fn try_new<A: AsRef<str>>(id: A) -> Result<Self, ShelfIdError> {
        let id_str = id.as_ref();
        let len = id_str.len();

        if len < 2 {
            return Err(ShelfIdError::TooShort);
        }
        if len > 64 {
            return Err(ShelfIdError::TooLong);
        }
        if !id_str
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(ShelfIdError::InvalidCharacters);
        }

        Ok(Self(id_str.to_string()))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for ShelfId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Shelf<DocId> {
    pub id: ShelfId,
    pub doc_ids: Vec<DocId>,
}

impl<DocId> Shelf<DocId> {
    pub async fn convert_ids<F, NewId>(self, f: F) -> Shelf<NewId>
    where
        F: Fn(DocId) -> NewId,
    {
        Shelf {
            id: self.id,
            doc_ids: self.doc_ids.into_iter().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shelf_id() {
        assert!(ShelfId::try_new("my-shelf").is_ok());
        assert!(ShelfId::try_new("ab").is_ok());
        assert!(ShelfId::try_new("ABC-123").is_ok());

        assert_eq!(ShelfId::try_new("a").unwrap_err(), ShelfIdError::TooShort);
        assert_eq!(
            ShelfId::try_new("a".repeat(65)).unwrap_err(),
            ShelfIdError::TooLong
        );

        assert_eq!(
            ShelfId::try_new("my shelf").unwrap_err(),
            ShelfIdError::InvalidCharacters
        );
        assert_eq!(
            ShelfId::try_new("my@shelf").unwrap_err(),
            ShelfIdError::InvalidCharacters
        );
        assert_eq!(
            ShelfId::try_new("my.shelf").unwrap_err(),
            ShelfIdError::InvalidCharacters
        );
    }
}
