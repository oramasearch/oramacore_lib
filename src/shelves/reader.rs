use std::{fmt::Debug, path::PathBuf};

use crate::fs::*;
use anyhow::Context;
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use super::file_util::{get_shelf_file_name, is_shelf_file, remove_shelf_file};
use super::op::ShelfOperation;
use super::{Shelf, ShelfId};

#[derive(Error, Debug)]
pub enum ShelvesReaderError {
    #[error("Io error {0:?}")]
    Io(std::io::Error),
    #[error("generic {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct ShelvesReader<DocumentId> {
    shelves: Vec<Shelf<DocumentId>>,
    shelf_ids_to_delete: Vec<ShelfId>,
}

impl<DocumentId: Serialize + DeserializeOwned + Debug + Clone> ShelvesReader<DocumentId> {
    pub fn empty() -> Self {
        Self {
            shelves: Vec::new(),
            shelf_ids_to_delete: Vec::new(),
        }
    }

    pub fn try_new(data_dir: PathBuf) -> Result<Self, ShelvesReaderError> {
        create_if_not_exists(&data_dir)?;

        let shelves: Vec<_> = std::fs::read_dir(&data_dir)
            .context("Cannot read shelf directory")?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if is_shelf_file(&path) {
                    let shelf: Shelf<DocumentId> =
                        BufferedFile::open(path).ok()?.read_json_data().ok()?;
                    Some(shelf)
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            shelves,
            shelf_ids_to_delete: Vec::new(),
        })
    }

    pub fn update(&mut self, op: ShelfOperation<DocumentId>) -> Result<(), ShelvesReaderError> {
        match op {
            ShelfOperation::Insert(shelf) => {
                self.shelf_ids_to_delete.retain(|name| name != &shelf.id);
                self.shelves.retain(|s| s.id != shelf.id);
                self.shelves.push(shelf);
            }
            ShelfOperation::Delete(shelf_name) => {
                self.shelves.retain(|s| s.id != shelf_name);
                self.shelf_ids_to_delete.push(shelf_name);
            }
        }
        Ok(())
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), ShelvesReaderError> {
        create_if_not_exists(&data_dir)?;

        for shelf in &self.shelves {
            let file_path = data_dir.join(get_shelf_file_name(shelf.id.as_str()));
            BufferedFile::create_or_overwrite(file_path)
                .context("Cannot create file")?
                .write_json_data(shelf)
                .context("Cannot write shelf to file")?;
        }

        for shelf_name_to_remove in self.shelf_ids_to_delete.drain(..) {
            let file_path = data_dir.join(get_shelf_file_name(shelf_name_to_remove.as_str()));
            remove_shelf_file(file_path);
        }

        Ok(())
    }

    pub fn get_shelf(&self, name: &ShelfId) -> Option<&Shelf<DocumentId>> {
        self.shelves.iter().find(|s| &s.id == name)
    }

    pub fn list_shelves(&self) -> &[Shelf<DocumentId>] {
        &self.shelves
    }
}

#[cfg(test)]
mod shelf_reader_tests {
    use super::*;

    #[test]
    fn test_shelf_reader_empty() {
        let reader: ShelvesReader<String> = ShelvesReader::empty();
        let shelves = reader.list_shelves();
        assert!(shelves.is_empty());
    }

    #[test]
    fn test_commit_shelves() {
        let base_dir = generate_new_path();
        let mut reader: ShelvesReader<String> = ShelvesReader::empty();

        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf-1").unwrap(),
                doc_ids: vec!["doc1".to_string(), "doc2".to_string()],
            }))
            .expect("Failed to insert shelf");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-1");
        assert_eq!(shelves[0].doc_ids.len(), 2);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let mut reader: ShelvesReader<String> =
            ShelvesReader::try_new(base_dir.clone()).expect("Failed to create ShelfReader");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-1");
        assert_eq!(shelves[0].doc_ids.len(), 2);

        reader
            .update(ShelfOperation::Delete(
                ShelfId::try_new("test-shelf-1").unwrap(),
            ))
            .expect("Failed to delete shelf");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 0);

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let reader: ShelvesReader<String> =
            ShelvesReader::try_new(base_dir.clone()).expect("Failed to create ShelfReader");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 0);
    }

    #[test]
    fn test_get_shelf() {
        let mut reader: ShelvesReader<usize> = ShelvesReader::empty();

        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf-1").unwrap(),
                doc_ids: vec![1, 2, 3],
            }))
            .expect("Failed to insert shelf");

        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf-2").unwrap(),
                doc_ids: vec![4, 5, 6],
            }))
            .expect("Failed to insert shelf");

        let shelf1 = reader.get_shelf(&ShelfId::try_new("test-shelf-1").unwrap());
        assert!(shelf1.is_some());
        assert_eq!(shelf1.unwrap().doc_ids, vec![1, 2, 3]);

        let shelf2 = reader.get_shelf(&ShelfId::try_new("test-shelf-2").unwrap());
        assert!(shelf2.is_some());
        assert_eq!(shelf2.unwrap().doc_ids, vec![4, 5, 6]);

        let shelf3 = reader.get_shelf(&ShelfId::try_new("non-existent").unwrap());
        assert!(shelf3.is_none());
    }

    #[test]
    fn test_update_existing_shelf() {
        let mut reader: ShelvesReader<String> = ShelvesReader::empty();

        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf").unwrap(),
                doc_ids: vec!["doc1".to_string()],
            }))
            .expect("Failed to insert shelf");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].doc_ids.len(), 1);

        // Update the same shelf with new documents
        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf").unwrap(),
                doc_ids: vec!["doc2".to_string(), "doc3".to_string()],
            }))
            .expect("Failed to update shelf");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].doc_ids.len(), 2);
        assert_eq!(shelves[0].doc_ids[0], "doc2");
        assert_eq!(shelves[0].doc_ids[1], "doc3");
    }

    #[test]
    fn test_delete_shelf_prevents_deletion_on_reinsert() {
        let base_dir = generate_new_path();
        let mut reader: ShelvesReader<String> = ShelvesReader::empty();

        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf").unwrap(),
                doc_ids: vec!["doc1".to_string()],
            }))
            .expect("Failed to insert shelf");

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let mut reader: ShelvesReader<String> =
            ShelvesReader::try_new(base_dir.clone()).expect("Failed to create ShelfReader");

        // Mark for deletion
        reader
            .update(ShelfOperation::Delete(
                ShelfId::try_new("test-shelf").unwrap(),
            ))
            .expect("Failed to delete shelf");

        // But then re-insert before commit
        reader
            .update(ShelfOperation::Insert(Shelf {
                id: ShelfId::try_new("test-shelf").unwrap(),
                doc_ids: vec!["doc2".to_string()],
            }))
            .expect("Failed to re-insert shelf");

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let reader: ShelvesReader<String> =
            ShelvesReader::try_new(base_dir.clone()).expect("Failed to create ShelfReader");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].doc_ids[0], "doc2");
    }

    #[test]
    fn test_multiple_shelves_persistence() {
        let base_dir = generate_new_path();
        let mut reader: ShelvesReader<usize> = ShelvesReader::empty();

        // Insert multiple shelves
        for i in 0..5 {
            reader
                .update(ShelfOperation::Insert(Shelf {
                    id: ShelfId::try_new(format!("shelf-{i}")).unwrap(),
                    doc_ids: vec![i, i + 10, i + 20],
                }))
                .expect("Failed to insert shelf");
        }

        reader
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let reader: ShelvesReader<usize> =
            ShelvesReader::try_new(base_dir.clone()).expect("Failed to create ShelfReader");

        let shelves = reader.list_shelves();
        assert_eq!(shelves.len(), 5);

        for i in 0..5 {
            let shelf_name = ShelfId::try_new(format!("shelf-{i}")).unwrap();
            let shelf = reader.get_shelf(&shelf_name);
            assert!(shelf.is_some());
            assert_eq!(shelf.unwrap().doc_ids.len(), 3);
        }
    }
}
