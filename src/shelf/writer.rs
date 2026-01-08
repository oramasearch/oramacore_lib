use super::file_util::{get_shelf_file_name, is_shelf_file, remove_shelf_file};
use super::{Shelf, ShelfId};
use crate::fs::{BufferedFile, create_if_not_exists};
use anyhow::Context;
use debug_panic::debug_panic;
use std::path::PathBuf;
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum ShelfWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct ShelfWriter {
    shelf: Vec<Shelf<String>>,
    shelf_ids_to_delete: Vec<ShelfId>,
    has_uncommitted_changes: bool,
}

impl ShelfWriter {
    pub fn empty() -> Result<Self, ShelfWriterError> {
        Ok(Self {
            shelf: Vec::new(),
            shelf_ids_to_delete: Vec::new(),
            has_uncommitted_changes: false,
        })
    }

    pub fn try_new(data_dir: PathBuf) -> Result<Self, ShelfWriterError> {
        create_if_not_exists(&data_dir)?;

        let dir = std::fs::read_dir(data_dir).context("Cannot read dir")?;

        let mut shelves = Vec::new();
        for entry in dir {
            let Ok(entry) = entry else {
                debug_panic!("This shouldn't happen");
                error!("Error occurred while trying to read shelf entry. Shelf skipped");
                continue;
            };

            if is_shelf_file(&entry.path()) {
                let shelf = BufferedFile::open(entry.path())
                    .context("cannot open shelf file")?
                    .read_json_data()
                    .context("cannot read shelf file")?;
                shelves.push(shelf);
            }
        }

        Ok(Self {
            shelf: shelves,
            shelf_ids_to_delete: Vec::new(),
            has_uncommitted_changes: false,
        })
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), ShelfWriterError> {
        create_if_not_exists(&data_dir)?;

        for s in &self.shelf {
            let file_path = data_dir.join(get_shelf_file_name(s.id.as_str()));
            BufferedFile::create_or_overwrite(file_path)
                .context("Cannot create file")?
                .write_json_data(s)
                .context("Cannot write shelf to file")?;
        }

        for shelf_id_to_remove in self.shelf_ids_to_delete.drain(..) {
            let file_path = data_dir.join(get_shelf_file_name(shelf_id_to_remove.as_str()));
            remove_shelf_file(file_path);
        }

        self.has_uncommitted_changes = false;

        Ok(())
    }

    pub fn insert_shelf(&mut self, shelf: Shelf<String>) -> Result<(), ShelfWriterError> {
        self.shelf.retain(|r| r.id != shelf.id);
        self.shelf_ids_to_delete.retain(|id| id != &shelf.id);
        self.shelf.push(shelf);
        self.has_uncommitted_changes = true;

        Ok(())
    }

    pub fn delete_shelf(&mut self, id: ShelfId) -> Result<(), ShelfWriterError> {
        self.shelf.retain(|r| r.id != id);
        self.shelf_ids_to_delete.push(id);
        self.has_uncommitted_changes = true;

        Ok(())
    }

    pub fn list_shelves(&self) -> &[Shelf<String>] {
        &self.shelf
    }

    pub fn has_pending_changes(&self) -> bool {
        self.has_uncommitted_changes
    }
}

#[cfg(test)]
mod shelf_tests {
    use super::*;
    use crate::fs::generate_new_path;

    #[tokio::test]
    async fn test_simple() {
        let path = generate_new_path();

        let mut writer = ShelfWriter::empty().unwrap();

        writer
            .insert_shelf(Shelf {
                id: ShelfId::try_new("test-shelf-1").unwrap(),
                documents: vec!["doc1".to_string(), "doc2".to_string()],
            })
            .await
            .unwrap();

        writer
            .insert_shelf(Shelf {
                id: ShelfId::try_new("test-shelf-2").unwrap(),
                documents: vec!["doc3".to_string()],
            })
            .await
            .unwrap();

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 2);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-1");
        assert_eq!(shelves[1].id.as_str(), "test-shelf-2");

        writer
            .delete_shelf(ShelfId::try_new("test-shelf-1").unwrap())
            .await
            .unwrap();

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-2");

        writer.commit(path.clone()).unwrap();

        let writer = ShelfWriter::try_new(path).unwrap();

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-2");
    }

    #[tokio::test]
    async fn test_commit_shelves() {
        let base_dir = generate_new_path();
        let mut writer = ShelfWriter::empty().unwrap();

        writer
            .insert_shelf(Shelf {
                id: ShelfId::try_new("test-shelf-1").unwrap(),
                documents: vec!["doc1".to_string()],
            })
            .await
            .expect("Failed to insert shelf");

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-1");

        writer
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let mut writer =
            ShelfWriter::try_new(base_dir.clone()).expect("Failed to create ShelfWriter");

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].id.as_str(), "test-shelf-1");

        writer
            .delete_shelf(ShelfId::try_new("test-shelf-1").unwrap())
            .await
            .expect("Failed to delete shelf");

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 0);

        writer
            .commit(base_dir.clone())
            .expect("Failed to commit shelves");

        let writer = ShelfWriter::try_new(base_dir.clone()).expect("Failed to create ShelfWriter");

        let shelves = writer.list_shelves();
        assert_eq!(shelves.len(), 0);
    }

    #[tokio::test]
    async fn test_has_pending_changes() {
        let path = generate_new_path();

        let mut writer = ShelfWriter::empty().unwrap();

        assert!(!writer.has_pending_changes());

        writer
            .insert_shelf(Shelf {
                id: ShelfId::try_new("test-shelf-1").unwrap(),
                documents: vec!["doc1".to_string()],
            })
            .await
            .unwrap();
        assert!(writer.has_pending_changes());

        writer.commit(path.clone()).unwrap();
        assert!(!writer.has_pending_changes());

        writer
            .delete_shelf(ShelfId::try_new("test-shelf-2").unwrap())
            .await
            .unwrap();
        assert!(writer.has_pending_changes());

        writer.commit(path.clone()).unwrap();
        assert!(!writer.has_pending_changes());
    }
}
