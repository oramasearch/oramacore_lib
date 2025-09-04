use std::{
    fmt::Debug,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use atomic_write_file::AtomicWriteFile;
use tracing::{error, trace};

pub async fn create_if_not_exists_async<P: AsRef<Path>>(p: P) -> Result<()> {
    let p: PathBuf = p.as_ref().to_path_buf();

    let output = tokio::fs::try_exists(&p).await;
    match output {
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Error while checking if the directory exists: {:?}",
                e
            ));
        }
        Ok(true) => {
            trace!("Directory exists. Skip creation.");
        }
        Ok(false) => {
            trace!("Directory does not exist. Creating it.");
            tokio::fs::create_dir_all(p)
                .await
                .context("Cannot create directory")?;
        }
    }

    Ok(())
}

pub fn create_if_not_exists<P: AsRef<Path>>(p: P) -> Result<()> {
    let p: PathBuf = p.as_ref().to_path_buf();

    match std::fs::exists(&p) {
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Error while checking if the directory exists: {:?}",
                e
            ));
        }
        Ok(true) => {
            trace!("Directory exists. Skip creation.");
        }
        Ok(false) => {
            trace!("Directory does not exist. Creating it.");
            std::fs::create_dir_all(p).context("Cannot create directory")?;
        }
    };

    Ok(())
}

pub async fn read_file<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T> {
    let vec = tokio::fs::read(&path)
        .await
        .with_context(|| format!("Cannot open file at {path:?}"))?;
    serde_json::from_slice(&vec)
        .with_context(|| format!("Cannot deserialize json data from {path:?}"))
}

pub struct BufferedFile;
impl BufferedFile {
    pub fn exists_as_file(path: &PathBuf) -> bool {
        std::fs::metadata(path)
            .map(|m| m.is_file())
            .unwrap_or(false)
    }

    pub fn create_or_overwrite(path: PathBuf) -> Result<WriteBufferedFile> {
        let buf = AtomicWriteFile::open(&path)
            .with_context(|| format!("Cannot create file at {path:?}"))?;
        // let file = std::fs::File::create(&path)
        //     .with_context(|| format!("Cannot create file at {:?}", path))?;
        let buf = BufWriter::new(buf);
        let buf = Some(buf);
        Ok(WriteBufferedFile { path, buf })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<ReadBufferedFile> {
        Ok(ReadBufferedFile {
            path: path.as_ref().to_path_buf(),
        })
    }
}

pub struct ReadBufferedFile {
    path: PathBuf,
}

impl ReadBufferedFile {
    pub fn read_json_data<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Cannot open file at {:?}", self.path))?;
        let reader = std::io::BufReader::new(file);
        let data = serde_json::from_reader(reader)
            .with_context(|| format!("Cannot read json data from {:?}", self.path))?;
        Ok(data)
    }

    pub fn read_text_data(self) -> Result<String> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Cannot open file at {:?}", self.path))?;
        let mut reader = std::io::BufReader::new(file);
        let mut b = String::new();
        reader
            .read_to_string(&mut b)
            .with_context(|| format!("Cannot read text data from {:?}", self.path))?;
        Ok(b)
    }

    pub fn read_bincode_data<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Cannot open file at {:?}", self.path))?;
        let reader = std::io::BufReader::new(file);
        let data = bincode::deserialize_from(reader)
            .with_context(|| format!("Cannot read bincode data from {:?}", self.path))?;
        Ok(data)
    }
}

pub struct WriteBufferedFile {
    path: PathBuf,
    buf: Option<BufWriter<AtomicWriteFile>>,
}
impl WriteBufferedFile {
    pub fn write_json_data<T: serde::Serialize + Debug>(mut self, data: &T) -> Result<()> {
        if let Some(buf) = self.buf.as_mut() {
            serde_json::to_writer(buf, data)
                .with_context(|| format!("Cannot write json data to {:?}", self.path))?;
        }

        self.close()
    }

    pub fn write_text_data<T: AsRef<[u8]>>(mut self, data: &T) -> Result<()> {
        if let Some(buf) = self.buf.as_mut() {
            buf.write_all(data.as_ref())
                .with_context(|| format!("Cannot write text data to {:?}", self.path))?;
        }

        self.close()
    }

    pub fn write_bincode_data<T: serde::Serialize + Debug>(mut self, data: &T) -> Result<()> {
        if let Some(buf) = self.buf.as_mut() {
            bincode::serialize_into(buf, data)
                .with_context(|| format!("Cannot write bincode data to {:?}", self.path))?;
        }

        self.close()
    }

    pub fn close(mut self) -> Result<()> {
        self.drop_all()
    }

    fn drop_all(&mut self) -> Result<()> {
        let mut buf = match self.buf.take() {
            Some(buf) => buf,
            None => {
                return Ok(());
            }
        };

        buf.flush()
            .with_context(|| format!("Cannot flush buffer {:?}", self.path))?;

        let buf = buf
            .into_inner()
            .with_context(|| format!("Cannot get inner buffer {:?}", self.path))?;
        buf.commit()
            .with_context(|| format!("Cannot commit buffer {:?}", self.path))?;

        // Be sure the file exists after commit
        // This is only for testing purposes, in production we don't want to check if the file exists
        if cfg!(any(test, debug_assertions)) {
            let exists = std::fs::read(&self.path)
                .with_context(|| format!("Cannot check if file exists {:?}", self.path))?;
            assert!(!exists.is_empty(), "File should exist after commit");
        }

        Ok(())
    }
}

// Proxy all std::io::Write methods to the inner buffer
impl std::io::Write for WriteBufferedFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(ref mut inner) = self.buf {
            inner.write(buf)
        } else {
            Err(std::io::Error::other("Buffered file is closed"))
        }
    }

    fn write_all(&mut self, mut buf: &[u8]) -> std::io::Result<()> {
        if let Some(ref mut inner) = self.buf {
            while !buf.is_empty() {
                let n = inner.write(buf)?;
                buf = &buf[n..];
            }
            Ok(())
        } else {
            Err(std::io::Error::other("Buffered file is closed"))
        }
    }

    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> std::io::Result<()> {
        if let Some(ref mut inner) = self.buf {
            inner.write_fmt(fmt)
        } else {
            Err(std::io::Error::other("Buffered file is closed"))
        }
    }

    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        if let Some(ref mut inner) = self.buf {
            inner.write_vectored(bufs)
        } else {
            Err(std::io::Error::other("Buffered file is closed"))
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(ref mut inner) = self.buf {
            inner.flush()
        } else {
            Err(std::io::Error::other("Buffered file is closed"))
        }
    }
}

impl Drop for WriteBufferedFile {
    fn drop(&mut self) {
        self.drop_all().unwrap_or_else(|e| {
            error!("Error while dropping buffered file: {:?}", e);
        });
    }
}

#[cfg(any(test, feature = "generate_new_path"))]
pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}
