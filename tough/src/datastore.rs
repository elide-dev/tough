// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{self, Result};
use jiff::Timestamp;
use log::debug;
use serde::Serialize;
use snafu::{ensure, ResultExt};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tempfile::TempDir;

/// Storage for persistent metadata used to detect rollback and clock rollback.
///
/// Names are repository metadata names, not paths chosen by the storage implementation. Implementors
/// must keep operations within their storage root and preserve existing data if a write fails.
/// Methods are synchronous, like the default filesystem implementation. Calls within a repository
/// are serialized by the datastore; implementations shared by repositories must also be thread safe.
pub trait DatastoreStorage: std::fmt::Debug + Send + Sync {
    /// Read one metadata object. Return `NotFound` when no object exists.
    fn read(&self, name: &str) -> std::io::Result<Vec<u8>>;
    /// Replace one metadata object with complete serialized bytes.
    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()>;
    /// Remove one metadata object. `NotFound` is accepted by the repository.
    fn remove(&self, name: &str) -> std::io::Result<()>;
}

/// `Datastore` persists TUF metadata files.
#[derive(Debug, Clone)]
pub(crate) struct Datastore {
    /// A lock around retrieving the datastore path.
    path_lock: Arc<RwLock<DatastorePath>>,
    /// A lock to treat the `system_time` function as a critical section.
    time_lock: Arc<Mutex<()>>,
}

impl Datastore {
    pub(crate) fn new(
        path: Option<PathBuf>,
        storage: Option<Arc<dyn DatastoreStorage>>,
    ) -> Result<Self> {
        Ok(Self {
            path_lock: Arc::new(RwLock::new(match (storage, path) {
                (None, None) => {
                    DatastorePath::TempDir(TempDir::new().context(error::DatastoreInitSnafu)?)
                }
                (None, Some(p)) => DatastorePath::Path(p),
                (Some(storage), _) => DatastorePath::Storage(storage),
            })),
            time_lock: Arc::new(Mutex::new(())),
        })
    }

    fn read_lock(&self) -> RwLockReadGuard<'_, DatastorePath> {
        self.path_lock.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_lock(&self) -> RwLockWriteGuard<'_, DatastorePath> {
        self.path_lock.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Get contents of a file in the datastore. This function is thread safe.
    ///
    /// TODO: [provide a thread safe interface](https://github.com/awslabs/tough/issues/602)
    ///
    pub(crate) async fn bytes(&self, file: &str) -> Result<Option<Vec<u8>>> {
        let lock = self.read_lock();
        let path = lock.path(file);
        match lock.read(file) {
            Ok(file) => Ok(Some(file)),
            Err(err) => match err.kind() {
                ErrorKind::NotFound => Ok(None),
                _ => Err(err).context(error::DatastoreOpenSnafu { path: &path }),
            },
        }
    }

    /// Writes a JSON metadata file in the datastore. This function is thread safe.
    pub(crate) async fn create<T: Serialize>(&self, file: &str, value: &T) -> Result<()> {
        let lock = self.write_lock();
        let path = lock.path(file);
        let bytes = serde_json::to_vec(value).with_context(|_| error::DatastoreSerializeSnafu {
            what: format!("{file} in datastore"),
            path: path.clone(),
        })?;
        lock.write(file, &bytes)
            .context(error::DatastoreCreateSnafu { path: &path })
    }

    /// Deletes a file from the datastore. This function is thread safe.
    pub(crate) async fn remove(&self, file: &str) -> Result<()> {
        let lock = self.write_lock();
        let path = lock.path(file);
        debug!("removing '{}'", path.display());
        match lock.remove(file) {
            Ok(()) => Ok(()),
            Err(err) => match err.kind() {
                ErrorKind::NotFound => Ok(()),
                _ => Err(err).context(error::DatastoreRemoveSnafu { path: &path }),
            },
        }
    }

    /// Ensures that system time has not stepped backward since it was last sampled. This function
    /// is protected by a lock guard to ensure thread safety.
    pub(crate) async fn system_time(&self) -> Result<Timestamp> {
        // Treat this function as a critical section. This lock is not used for anything else.
        let lock = self.time_lock.lock().unwrap_or_else(|e| e.into_inner());

        let file = "latest_known_time.json";
        // Load the latest known system time, if it exists
        let poss_latest_known_time = self
            .bytes(file)
            .await?
            .map(|b| serde_json::from_slice::<Timestamp>(&b));

        // Get 'current' system time
        let sys_time = Timestamp::now();

        if let Some(Ok(latest_known_time)) = poss_latest_known_time {
            // Make sure the sampled system time did not go back in time
            ensure!(
                sys_time >= latest_known_time,
                error::SystemTimeSteppedBackwardSnafu {
                    sys_time,
                    latest_known_time
                }
            );
        }
        // Store the latest known time
        // Serializes RFC3339 time string and store to datastore
        self.create(file, &sys_time).await?;

        // Explicitly drop the lock to avoid any compiler optimization.
        drop(lock);
        Ok(sys_time)
    }
}

/// Because `TempDir` is an RAII object, we need to hold on to it. This private enum allows us to
/// hold either a `TempDir` or a `PathBuf` depending on whether or not the user wants to manage the
/// directory.
#[derive(Debug)]
enum DatastorePath {
    /// Path to a user-managed directory.
    Path(PathBuf),
    /// A `TempDir` that we created on the user's behalf.
    TempDir(TempDir),
    /// Storage supplied by the application, retaining its own authority and lifetime.
    Storage(Arc<dyn DatastoreStorage>),
}

impl DatastorePath {
    /// Location used in diagnostics; custom storage does not expose an ambient pathname.
    fn path(&self, file: &str) -> PathBuf {
        match self {
            DatastorePath::Path(p) => p.join(file),
            DatastorePath::TempDir(t) => t.path().join(file),
            DatastorePath::Storage(_) => PathBuf::from(file),
        }
    }

    fn read(&self, file: &str) -> std::io::Result<Vec<u8>> {
        match self {
            Self::Storage(storage) => storage.read(file),
            _ => std::fs::read(self.path(file)),
        }
    }

    fn write(&self, file: &str, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            Self::Storage(storage) => storage.write(file, bytes),
            _ => std::fs::write(self.path(file), bytes),
        }
    }

    fn remove(&self, file: &str) -> std::io::Result<()> {
        match self {
            Self::Storage(storage) => storage.remove(file),
            _ => std::fs::remove_file(self.path(file)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug, Default)]
    struct Memory(Mutex<HashMap<String, Vec<u8>>>);

    impl DatastoreStorage for Memory {
        fn read(&self, name: &str) -> std::io::Result<Vec<u8>> {
            self.0
                .lock()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(|| ErrorKind::NotFound.into())
        }
        fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(name.to_owned(), bytes.to_vec());
            Ok(())
        }
        fn remove(&self, name: &str) -> std::io::Result<()> {
            self.0
                .lock()
                .unwrap()
                .remove(name)
                .map(|_| ())
                .ok_or_else(|| ErrorKind::NotFound.into())
        }
    }

    #[tokio::test]
    async fn application_storage_preserves_metadata_and_clock_rollback_checks() {
        let storage = Arc::new(Memory::default());
        let first = Datastore::new(None, Some(storage.clone())).unwrap();
        assert_eq!(first.bytes("timestamp.json").await.unwrap(), None);
        first
            .create("timestamp.json", &serde_json::json!({"version": 2}))
            .await
            .unwrap();
        drop(first);
        let reopened = Datastore::new(None, Some(storage.clone())).unwrap();
        assert_eq!(
            reopened.bytes("timestamp.json").await.unwrap().unwrap(),
            br#"{"version":2}"#
        );
        reopened.remove("timestamp.json").await.unwrap();
        reopened.remove("timestamp.json").await.unwrap();
        assert_eq!(reopened.bytes("timestamp.json").await.unwrap(), None);
        let future = Timestamp::from_second(4_102_444_800).unwrap();
        reopened
            .create("latest_known_time.json", &future)
            .await
            .unwrap();
        assert!(matches!(
            reopened.system_time().await,
            Err(error::Error::SystemTimeSteppedBackward { .. })
        ));
        assert_eq!(
            serde_json::from_slice::<Timestamp>(&storage.read("latest_known_time.json").unwrap())
                .unwrap(),
            future
        );
    }

    #[derive(Debug)]
    struct Denied;
    impl DatastoreStorage for Denied {
        fn read(&self, _: &str) -> std::io::Result<Vec<u8>> {
            Err(ErrorKind::PermissionDenied.into())
        }
        fn write(&self, _: &str, _: &[u8]) -> std::io::Result<()> {
            Err(ErrorKind::PermissionDenied.into())
        }
        fn remove(&self, _: &str) -> std::io::Result<()> {
            Err(ErrorKind::PermissionDenied.into())
        }
    }

    #[tokio::test]
    async fn application_storage_errors_never_fall_back_to_an_ambient_directory() {
        let path = tempfile::tempdir().unwrap();
        std::fs::write(path.path().join("timestamp.json"), b"keep").unwrap();
        let datastore =
            Datastore::new(Some(path.path().to_owned()), Some(Arc::new(Denied))).unwrap();
        assert!(datastore.bytes("timestamp.json").await.is_err());
        assert!(datastore.create("timestamp.json", &0).await.is_err());
        assert!(datastore.remove("timestamp.json").await.is_err());
        assert_eq!(
            std::fs::read(path.path().join("timestamp.json")).unwrap(),
            b"keep"
        );
    }
}
