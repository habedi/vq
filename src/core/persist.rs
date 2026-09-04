//! Byte and file persistence shared by all quantizers.
//!
//! Models are encoded with the postcard format through serde. The methods are added to
//! each quantizer type by [`impl_persist!`], which also runs the type's `validate` step
//! after decoding so that malformed bytes are rejected instead of producing a quantizer
//! that breaks its own invariants.

use crate::core::error::{VqError, VqResult};
use serde::Serialize;
use serde::de::DeserializeOwned;

pub(crate) fn to_bytes<T: Serialize>(value: &T) -> VqResult<Vec<u8>> {
    postcard::to_stdvec(value).map_err(|e| VqError::Serialization(e.to_string()))
}

pub(crate) fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> VqResult<T> {
    postcard::from_bytes(bytes).map_err(|e| VqError::Serialization(e.to_string()))
}

/// Adds `to_bytes`, `from_bytes`, `save`, and `load` to a serializable quantizer type.
///
/// The type must provide `fn validate(self) -> VqResult<Self>`.
macro_rules! impl_persist {
    ($ty:ty) => {
        impl $ty {
            /// Encodes the quantizer into a compact byte representation.
            ///
            /// # Errors
            ///
            /// Returns an error if encoding fails.
            pub fn to_bytes(&self) -> $crate::core::error::VqResult<Vec<u8>> {
                $crate::core::persist::to_bytes(self)
            }

            /// Decodes a quantizer previously produced by [`to_bytes`](Self::to_bytes).
            ///
            /// # Errors
            ///
            /// Returns an error if the bytes are malformed or describe an invalid quantizer.
            pub fn from_bytes(bytes: &[u8]) -> $crate::core::error::VqResult<Self> {
                let decoded: Self = $crate::core::persist::from_bytes(bytes)?;
                decoded.validate()
            }

            /// Writes the quantizer to a file.
            ///
            /// # Errors
            ///
            /// Returns an error if encoding or writing fails.
            pub fn save<P: AsRef<std::path::Path>>(
                &self,
                path: P,
            ) -> $crate::core::error::VqResult<()> {
                let bytes = self.to_bytes()?;
                std::fs::write(path, bytes)?;
                Ok(())
            }

            /// Reads a quantizer from a file written by [`save`](Self::save).
            ///
            /// # Errors
            ///
            /// Returns an error if reading or decoding fails.
            pub fn load<P: AsRef<std::path::Path>>(path: P) -> $crate::core::error::VqResult<Self> {
                let bytes = std::fs::read(path)?;
                Self::from_bytes(&bytes)
            }
        }
    };
}

pub(crate) use impl_persist;
