use super::{HookOperation, HookType};
use crate::fs::*;
use std::future::Future;
use std::pin::Pin;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HookWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

// Type alias for the hook operation callback
pub type HookOperationCallback =
    Box<dyn Fn(HookOperation) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub struct HookWriter {
    base_dir: PathBuf,
    before_retrieval_presence: AtomicBool,
    before_answer_presence: AtomicBool,
    before_search_presence: AtomicBool,
    transform_document_before_save_presence: AtomicBool,
    transform_document_after_search_presence: AtomicBool,
    f: HookOperationCallback,
}

impl HookWriter {
    pub fn try_new(base_dir: PathBuf, f: HookOperationCallback) -> Result<Self, HookWriterError> {
        create_if_not_exists(&base_dir)?;

        let before_retrieval_file = base_dir.join(HookType::BeforeRetrieval.get_file_name());
        let before_retrieval_presence = BufferedFile::exists_as_file(&before_retrieval_file);

        let before_answer_file = base_dir.join(HookType::BeforeAnswer.get_file_name());
        let before_answer_presence = BufferedFile::exists_as_file(&before_answer_file);

        let before_search_file = base_dir.join(HookType::BeforeSearch.get_file_name());
        let before_search_presence = BufferedFile::exists_as_file(&before_search_file);

        let transform_document_before_save_file =
            base_dir.join(HookType::TransformDocumentBeforeSave.get_file_name());
        let transform_document_before_save_presence =
            BufferedFile::exists_as_file(&transform_document_before_save_file);

        let transform_document_after_search_file =
            base_dir.join(HookType::TransformDocumentAfterSearch.get_file_name());
        let transform_document_after_search_presence =
            BufferedFile::exists_as_file(&transform_document_after_search_file);

        Ok(Self {
            base_dir,
            before_retrieval_presence: AtomicBool::new(before_retrieval_presence),
            before_answer_presence: AtomicBool::new(before_answer_presence),
            before_search_presence: AtomicBool::new(before_search_presence),
            transform_document_before_save_presence: AtomicBool::new(
                transform_document_before_save_presence,
            ),
            transform_document_after_search_presence: AtomicBool::new(
                transform_document_after_search_presence,
            ),
            f,
        })
    }

    pub async fn insert_hook(
        &self,
        hook_type: HookType,
        code: String,
    ) -> Result<(), HookWriterError> {
        match hook_type {
            HookType::BeforeRetrieval => {
                self.before_retrieval_presence
                    .store(true, Ordering::Relaxed);
            }
            HookType::BeforeAnswer => {
                self.before_answer_presence.store(true, Ordering::Relaxed);
            }
            HookType::BeforeSearch => {
                self.before_search_presence.store(true, Ordering::Relaxed);
            }
            HookType::TransformDocumentBeforeSave => {
                self.transform_document_before_save_presence
                    .store(true, Ordering::Relaxed);
            }
            HookType::TransformDocumentAfterSearch => {
                self.transform_document_after_search_presence
                    .store(true, Ordering::Relaxed);
            }
        };
        let path = self.base_dir.join(hook_type.get_file_name());
        BufferedFile::create_or_overwrite(path)?.write_text_data(&code)?;

        (self.f)(HookOperation::Insert(hook_type, code)).await;

        Ok(())
    }

    pub async fn delete_hook(&self, hook_type: HookType) -> Result<(), HookWriterError> {
        let path = self.base_dir.join(hook_type.get_file_name());
        match std::fs::remove_file(&path) {
            Ok(_) => match hook_type {
                HookType::BeforeRetrieval => {
                    self.before_retrieval_presence
                        .store(false, Ordering::Relaxed);
                }
                HookType::BeforeAnswer => {
                    self.before_answer_presence.store(false, Ordering::Relaxed);
                }
                HookType::BeforeSearch => {
                    self.before_search_presence.store(false, Ordering::Relaxed);
                }
                HookType::TransformDocumentBeforeSave => {
                    self.transform_document_before_save_presence
                        .store(false, Ordering::Relaxed);
                }
                HookType::TransformDocumentAfterSearch => {
                    self.transform_document_after_search_presence
                        .store(false, Ordering::Relaxed);
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File does not exist, treat as success
            }
            Err(e) => return Err(HookWriterError::FSError(e)),
        };

        (self.f)(HookOperation::Delete(hook_type)).await;

        Ok(())
    }

    pub fn list_hooks(&self) -> Result<Vec<(HookType, Option<String>)>, HookWriterError> {
        let types = vec![
            HookType::BeforeRetrieval,
            HookType::BeforeAnswer,
            HookType::BeforeSearch,
            HookType::TransformDocumentBeforeSave,
            HookType::TransformDocumentAfterSearch,
        ];

        let mut ret = Vec::with_capacity(types.len());
        for hook_type in types {
            let path = self.base_dir.join(hook_type.get_file_name());

            let content = BufferedFile::open(path)
                .and_then(|f| f.read_text_data())
                .ok();
            ret.push((hook_type, content));
        }

        Ok(ret)
    }

    pub fn get_hook_content(&self, hook_type: HookType) -> Result<Option<String>, HookWriterError> {
        let path = self.base_dir.join(hook_type.get_file_name());

        let content = BufferedFile::open(path)
            .and_then(|f| f.read_text_data())
            .ok();

        Ok(content)
    }

    pub fn has_hook(&self, hook_type: HookType) -> bool {
        match hook_type {
            HookType::BeforeRetrieval => self.before_retrieval_presence.load(Ordering::Relaxed),
            HookType::BeforeAnswer => self.before_answer_presence.load(Ordering::Relaxed),
            HookType::BeforeSearch => self.before_search_presence.load(Ordering::Relaxed),
            HookType::TransformDocumentBeforeSave => self
                .transform_document_before_save_presence
                .load(Ordering::Relaxed),
            HookType::TransformDocumentAfterSearch => self
                .transform_document_after_search_presence
                .load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use std::sync::{Arc, RwLock};

    #[tokio::test]
    async fn test_hook_writer_lifecycle() {
        let base_dir = generate_new_path();

        // Shared vector to record HookOperation invocations
        let ops: Arc<RwLock<Vec<HookOperation>>> = Arc::new(RwLock::new(Vec::new()));

        let ops_clone = ops.clone();
        let dummy_f = Box::new(move |op: HookOperation| {
            let ops_inner = ops_clone.clone();
            async move {
                ops_inner.write().unwrap().push(op);
            }
            .boxed()
        });

        let writer =
            HookWriter::try_new(base_dir.clone(), dummy_f).expect("Failed to create HookWriter");

        // Initially, no hook file should exist
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 5);
        assert!(hooks[0].1.is_none());
        assert!(hooks[1].1.is_none());
        assert!(hooks[2].1.is_none());
        assert!(hooks[3].1.is_none());
        assert!(hooks[4].1.is_none());

        // Insert a hook
        let code = r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }
"#
        .to_string();
        writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .expect("insert_hook failed");

        // list_hooks should return the code
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 5);
        assert_eq!(hooks[0].1.as_deref(), Some(code.as_str()));
        assert!(hooks[1].1.is_none());
        assert!(hooks[2].1.is_none());
        assert!(hooks[3].1.is_none());
        assert!(hooks[4].1.is_none());

        // Delete the hook
        writer
            .delete_hook(HookType::BeforeRetrieval)
            .await
            .expect("delete_hook failed");

        // list_hooks should return None for content
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 5);
        assert!(hooks[0].1.is_none());
        assert!(hooks[1].1.is_none());
        assert!(hooks[2].1.is_none());
        assert!(hooks[3].1.is_none());
        assert!(hooks[4].1.is_none());

        // Assert closure invocations at the end
        let ops = ops.read().unwrap();
        assert_eq!(ops.len(), 2);
        assert!(
            matches!(ops[0], HookOperation::Insert(HookType::BeforeRetrieval, ref c) if c == &code)
        );
        assert!(matches!(
            ops[1],
            HookOperation::Delete(HookType::BeforeRetrieval)
        ));
    }

    #[tokio::test]
    async fn test_hook_writer_persistency() {
        let base_dir = generate_new_path();

        let code = r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }
"#
        .to_string();

        let dummy_f = Box::new(move |_: HookOperation| async move {}.boxed());
        let writer = HookWriter::try_new(base_dir.clone(), dummy_f)
            .expect("Failed to create first HookWriter");

        writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .expect("insert_hook failed");

        // Create a new HookWriter with the same base_dir and verify persistence
        let dummy_f = Box::new(move |_: HookOperation| async move {}.boxed());
        let writer = HookWriter::try_new(base_dir.clone(), dummy_f)
            .expect("Failed to create second HookWriter");

        let content = writer
            .get_hook_content(HookType::BeforeRetrieval)
            .expect("get_hook_content failed");
        assert_eq!(content.as_deref(), Some(code.as_str()));
    }
}
