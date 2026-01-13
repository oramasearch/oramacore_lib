use std::path::{Path, PathBuf};
use tracing::error;

const SHELF_EXTENSION: &str = "shelf";

pub fn get_shelf_file_name(id: &str) -> String {
    format!("{id}.{SHELF_EXTENSION}")
}

pub fn is_shelf_file(p: &Path) -> bool {
    p.extension().and_then(|os| os.to_str()) == Some(SHELF_EXTENSION)
}

pub fn remove_shelf_file(p: PathBuf) {
    match std::fs::remove_file(&p) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            error!(error = ?e, "Cannot remove shelf file: {}", &p.display());
        }
    }
}
