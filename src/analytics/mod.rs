use std::{io::Write, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use chrono::Utc;
use futures::StreamExt;
use serde::Serialize;
use tokio::time::sleep;
use tokio_util::io::ReaderStream;
use tracing::{error, info};

use crate::fs::{BufferedFile, create_if_not_exists};

pub struct AnalyticConfig {
    pub data_dir: PathBuf,
}

pub struct AnalyticsStorage<Event> {
    data_dir: PathBuf,
    sender: tokio::sync::mpsc::Sender<InternalEvent<Event>>,
}

impl<Event: Serialize + Send + Sync + 'static> AnalyticsStorage<Event> {
    pub fn try_new(config: AnalyticConfig) -> Result<Self> {
        let data_dir = config.data_dir;
        create_if_not_exists(&data_dir)?;

        let init_file_name: String =
            if BufferedFile::exists_as_file(&data_dir.join("analytics.index")) {
                BufferedFile::open(data_dir.join("analytics.index"))
                    .with_context(|| {
                        format!("Cannot open analytics file at {}", data_dir.display())
                    })?
                    .read_text_data()?
            } else {
                // Default file name is based on the current timestamp
                let now = Utc::now().timestamp();
                format!("analytics_{now}.log")
            };

        BufferedFile::create_or_overwrite(data_dir.join("analytics.index"))
            .with_context(|| format!("Cannot open analytics file at {}", data_dir.display()))?
            .write_text_data(&init_file_name)?;

        let (sender, receiver) = tokio::sync::mpsc::channel::<InternalEvent<Event>>(100);

        tokio::task::spawn(store_event_loop(data_dir.clone(), receiver, init_file_name));

        Ok(Self { data_dir, sender })
    }

    pub fn add_event(&self, event: Event) -> Result<()> {
        let internal_event = InternalEvent::NewEvent(event);
        if let Err(e) = self.sender.try_send(internal_event) {
            error!(error = ?e, "Failed to send analytic event");
            return Err(anyhow::anyhow!("Failed to send analytic event"));
        }
        Ok(())
    }

    pub async fn get_and_erase(&self) -> Result<AnalyticLogStream> {
        let (sender, receiver) = tokio::sync::oneshot::channel::<(PathBuf, String)>();
        let internal_event = InternalEvent::Rotate(sender);
        if let Err(e) = self.sender.try_send(internal_event) {
            error!(error = ?e, "Failed to send rotate signal");
            return Err(anyhow::anyhow!("Failed to send rotate signal"));
        }

        let (previous_file_path, new_file_name) = receiver
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive rotate signal"))?;
        let file = tokio::fs::File::open(&previous_file_path).await?;

        BufferedFile::create_or_overwrite(self.data_dir.join("analytics.index"))
            .with_context(|| format!("Cannot open analytics file at {}", self.data_dir.display()))?
            .write_text_data(&new_file_name)?;

        let stream: ReaderStream<tokio::fs::File> = ReaderStream::new(file);
        let stream = AnalyticLogStream {
            file_path: previous_file_path.clone(),
            stream,
            already_deleted: false,
        };

        Ok(stream)
    }
}

pub struct AnalyticLogStream {
    file_path: PathBuf,
    stream: ReaderStream<tokio::fs::File>,
    already_deleted: bool,
}

impl futures::Stream for AnalyticLogStream {
    type Item = Result<tokio_util::bytes::Bytes, anyhow::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let ret = self.as_mut().stream.poll_next_unpin(cx);

        if matches!(ret, std::task::Poll::Ready(None)) && !self.already_deleted {
            self.already_deleted = true;
            if let Err(e) = std::fs::remove_file(&self.file_path) {
                error!(error = ?e, "Failed to delete analytics file");
            } else {
                info!("Analytics file {} deleted", self.file_path.display());
            }
        }

        ret.map(|opt| opt.map(|bytes| bytes.map_err(anyhow::Error::from)))
    }
}

async fn store_event_loop<Event: Serialize + Send + Sync>(
    base_dir: PathBuf,
    mut receiver: tokio::sync::mpsc::Receiver<InternalEvent<Event>>,
    init_file_name: String,
) -> Result<()> {
    let mut file_path = base_dir.join(init_file_name);

    let mut file = std::fs::File::options()
        .create(true)
        .write(true)
        .read(true)
        .truncate(false)
        .open(&file_path)
        .with_context(|| format!("Cannot open file at {file_path:?}"))?;

    // 100 is an arbitrary limit for the number of events to process at once
    let limit = 100;

    let mut buffer = Vec::with_capacity(limit);

    loop {
        let rec = receiver.recv_many(&mut buffer, limit).await;
        if rec == 0 {
            info!("No more events to process, exiting event loop");
            // Acording to the documentation, this means the channel is closed
            // because the limit is greater than 0 (it is hard-coded)
            break;
        }

        info!(count = rec, "Processing {} analytic events", rec);
        for event in buffer.drain(..) {
            match event {
                InternalEvent::NewEvent(ev) => {
                    write_to_file(&mut file, &ev)
                        .with_context(|| format!("Cannot write event to file at {file_path:?}"))?;
                }
                InternalEvent::Rotate(sender) => {
                    let previous_file_path = file_path;
                    let (new_file_path, file_name) = loop {
                        let now = Utc::now().timestamp();
                        let file_name = format!("analytics_{now}.log");
                        let new_file_path = base_dir.join(&file_name);
                        if new_file_path != previous_file_path {
                            break (new_file_path, file_name);
                        }
                        sleep(Duration::from_millis(100)).await; // Force a new timestamp
                    };
                    file_path = new_file_path;

                    file = std::fs::File::options()
                        .create(true)
                        .write(true)
                        .read(true)
                        .truncate(false)
                        .open(&file_path)
                        .with_context(|| format!("Cannot open file at {file_path:?}"))?;

                    info!(
                        "Analytics file rotated: old_file {previous_file_path:?} new_file {file_path:?}"
                    );

                    if let Err(e) = sender.send((previous_file_path, file_name)) {
                        error!(error = ?e, "Failed to send rotate signal");
                    }
                }
            }

            if let Err(e) = file.flush() {
                error!(error = ?e, "Failed to flush file");
            }
            if let Err(e) = file.sync_all() {
                error!(error = ?e, "Failed to sync file");
            }
        }
    }

    Ok(())
}

fn write_to_file<T: Serialize>(file: &mut std::fs::File, data: &T) -> Result<()> {
    let data = serde_json::to_string(data).context("Cannot serialize data to JSON")?;
    file.write_all(data.as_bytes())
        .context("Cannot write data to file")?;
    if let Err(e) = file.write(b"\n") {
        error!(error = ?e, "Failed to write newline to file");
    }
    Ok(())
}

enum InternalEvent<Event> {
    NewEvent(Event),
    Rotate(tokio::sync::oneshot::Sender<(PathBuf, String)>),
}

#[cfg(test)]
mod tests_analytics {
    use futures::{FutureExt, future::BoxFuture};
    use serde::Deserialize;

    use crate::fs::generate_new_path;

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct MyEvent {
        id: u64,
    }

    #[tokio::test]
    async fn test_analytics_storage_lifecycle() {
        let data_dir = generate_new_path();

        let storage = AnalyticsStorage::try_new(AnalyticConfig {
            data_dir: data_dir.clone(),
        })
        .unwrap();

        let create_event = |id: i64| MyEvent { id: id as u64 };
        storage.add_event(create_event(1)).unwrap();
        storage.add_event(create_event(2)).unwrap();

        let old_file: PathBuf = wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entries: Vec<_> = std::fs::read_dir(data_dir).unwrap().collect();
                if entries.len() == 2 {
                    // Only the new file should exist + the index file
                    let entry = entries
                        .into_iter()
                        .find(|e| e.as_ref().unwrap().file_name() != "analytics.index")
                        .ok_or_else(|| anyhow::anyhow!("No index file found"))??;
                    return Ok(entry.path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();
        let file_content = std::fs::read_to_string(&old_file).unwrap();
        assert_eq!(file_content.lines().count(), 2);

        let mut stream = storage.get_and_erase().await.unwrap();

        storage.add_event(create_event(3)).unwrap();

        wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entries: Vec<_> = std::fs::read_dir(data_dir)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()?;
                if entries.len() == 3 {
                    // The old file + new file should exist + the index file
                    return Ok(entries[2].path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();

        let mut stream_content = String::new();
        while let Some(Ok(bytes)) = stream.next().await {
            // Process the bytes as needed
            stream_content.push_str(&String::from_utf8_lossy(&bytes));
        }
        assert_eq!(
            stream_content.split('\n').count(),
            3,
            "There should be two lines in the analytics log + 1 for the newline"
        );
        assert!(stream_content.contains("\"id\":1"));
        assert!(stream_content.contains("\"id\":2"));

        let new_file: PathBuf = wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entry: Vec<_> = std::fs::read_dir(data_dir).unwrap().collect();
                if entry.len() == 2 {
                    // Only the new file should remain + the index file
                    return Ok(entry[1].as_ref().unwrap().path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();

        assert_ne!(
            old_file, new_file,
            "Old file should be different from new file"
        );

        // Simulate a server restart
        drop(storage);
        // Reload the storage with the same data directory
        let storage: AnalyticsStorage<MyEvent> = AnalyticsStorage::try_new(AnalyticConfig {
            data_dir: data_dir.clone(),
        })
        .unwrap();

        let mut stream = storage.get_and_erase().await.unwrap();
        let mut stream_content = String::new();
        while let Some(Ok(bytes)) = stream.next().await {
            // Process the bytes as needed
            stream_content.push_str(&String::from_utf8_lossy(&bytes));
        }
        assert_eq!(
            stream_content.split('\n').count(),
            2,
            "There should be one line in the analytics log + 1 for the newline"
        );
        assert!(!stream_content.contains("\"id\":1"));
        assert!(!stream_content.contains("\"id\":2"));
        assert!(stream_content.contains("\"id\":3"));
    }

    #[tokio::test]
    async fn test_analytics_storage_inner_write() {
        let data_dir = generate_new_path();
        create_if_not_exists(&data_dir).unwrap();
        let file_name = data_dir.join("analytics.log");

        let mut file = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&file_name)
            .unwrap();

        write_to_file(&mut file, &"test data").unwrap();
        write_to_file(&mut file, &"test data2").unwrap();

        let content = std::fs::read_to_string(&file_name)
            .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", file_name.display(), e))
            .unwrap();

        assert!(content.contains("test data"));
        assert!(content.contains("test data2"));
        assert!(content.lines().count() == 2);
    }

    pub async fn wait_for<'i, 'b, I, R>(
        i: &'i I,
        f: impl Fn(&'i I) -> BoxFuture<'b, Result<R>>,
    ) -> Result<R>
    where
        'b: 'i,
    {
        // 20 msec * 2_000 attempts = 40_000 msec = 40 sec
        const MAX_ATTEMPTS: usize = 2_000;
        let mut attempts = 0;
        loop {
            attempts += 1;
            match f(i).await {
                Ok(r) => break Ok(r),
                Err(e) => {
                    if attempts > MAX_ATTEMPTS {
                        break Err(e);
                    }
                    sleep(Duration::from_millis(20)).await
                }
            }
        }
    }
}
