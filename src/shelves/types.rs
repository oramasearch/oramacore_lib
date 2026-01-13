use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShelfIdError {
    #[error("Shelf id must be at least 2 characters")]
    TooShort,
    #[error("Shelf id must be at most 64 characters")]
    TooLong,
    #[error(
        "Shelf id must contain only alphanumeric characters, dashes, underscores or dollars signs"
    )]
    InvalidCharacters,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '$')
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

impl Serialize for ShelfId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ShelfId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        ShelfId::try_new(s).map_err(de::Error::custom)
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
        assert!(ShelfId::try_new("ABC-123$").is_ok());

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

    #[test]
    fn test_shelf_deserialization_validation() {
        let json = r#"{"id": "valid-shelf", "doc_ids": ["doc1", "doc2"]}"#;
        assert!(serde_json::from_str::<Shelf<String>>(json).is_ok());

        let json_invalid = r#"{"id": "a", "doc_ids": ["doc1", "doc2"]}"#;
        assert!(serde_json::from_str::<Shelf<String>>(json_invalid).is_err());

        let json_invalid_chars = r#"{"id": "invalid shelf!", "doc_ids": ["doc1", "doc2"]}"#;
        assert!(serde_json::from_str::<Shelf<String>>(json_invalid_chars).is_err());
    }
}
