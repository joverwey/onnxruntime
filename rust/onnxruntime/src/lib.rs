//-------------------------------------------------------------------------------------------------
// MACROS
//-------------------------------------------------------------------------------------------------
#[macro_use]

macro_rules! try_get_fn {
    ($api:expr, $function_name:ident) => {
        $api.$function_name
            .ok_or(OnnxError::ApiFunctionError(stringify!($function_name)))?
    };
}

macro_rules! invoke_fn {
    ($api:expr, $function_name:ident $(, $param:expr)*) => {
        let status = unsafe {$function_name($($param),*)};
        check_status($api, status)?
    };
}

macro_rules! try_invoke {
    ($api:expr, $function_name:ident $(, $param:expr)*) => {
        let f = try_get_fn!($api, $function_name);
        invoke_fn!($api, f, $($param),*)
    };
}

macro_rules! try_create {
    ($api:expr, $function_name:ident, $type_name:ty) => {

        {
            let mut t: *mut $type_name = std::ptr::null_mut();
            let t_ptr: *mut *mut $type_name = &mut t;
            try_invoke!($api, $function_name, t_ptr);
            t
        }

    };
    ($api:expr, $function_name:ident, $type_name:ty $(, $param:expr)*) => {

        {
            let mut t: *mut $type_name = std::ptr::null_mut();
            let t_ptr: *mut *mut $type_name = &mut t;
            try_invoke!($api, $function_name, $($param),*, t_ptr);
            t
        }

    };
}

//-------------------------------------------------------------------------------------------------
// IMPORTS
//-------------------------------------------------------------------------------------------------
pub mod node;
pub mod session;
pub mod shaped_data;
pub mod tensor;
pub mod tensor_element;

use onnxruntime_sys::{OrtApi, OrtGetApiBase, OrtStatus};

use session::{Session, SessionOptions};
use shaped_data::ShapedData;
use std::ffi::CStr;
use thiserror::Error;
//-------------------------------------------------------------------------------------------------
// CONSTANTS
//-------------------------------------------------------------------------------------------------
const ORT_API_VERSION: u32 = 2;

//-------------------------------------------------------------------------------------------------
// TYPES
//-------------------------------------------------------------------------------------------------
#[derive(Copy, Clone)]
pub enum LoggingLevel {
    Verbose,
    Info,
    Warning,
    Error,
    Fatal,
}

//-------------------------------------------------------------------------------------------------
///An error type to represent a bounded set of things that can go wrong while using this API.
#[derive(Error, Debug)]
pub enum OnnxError {
    #[error("The Api function pointer for '{0}' was null")]
    ApiFunctionError(&'static str),

    #[error("The input string '{0}' must be convertible to a c-string")]
    InvalidString(String),

    #[error("The error message returned by ONNX was not valid UTF8")]
    InvalidErrorMessage,

    #[error("ONNX Error: {0}")]
    Status(String),

    #[error("{0} was null")]
    NullPointer(&'static str),

    #[error("Expected tensor dimension count to be {0} but was {1}.")]
    InvalidTensorDimensionCount(usize, usize),

    #[error("Failed to create output tensor.")]
    NotATensor,

    #[error("Tensor type mismatch. Expected {0} but received. {1}")]
    TensorTypeMismatch(String, String),

    #[error("Tensor dimension mismatch. Expected {0} but received {1} dimensions")]
    TensorDimensionMismatch(usize, usize),

    #[error("The product of the all the tensor dimensions should equal the total number of elements in the flattened vector. The dimensions were {0} but the total number of elements were {1} ")]
    TensorDimensionElementCountMismatch(String, usize),

    #[error("Could not create tensor with shape {0:?}.")]
    InvalidTensorShape(Vec<i64>),
}

//-------------------------------------------------------------------------------------------------
/// The main API class used for most functions and object creation.
pub struct Onnx {
    api: OrtApi,
}

pub struct Opaque<T> {
    ptr: *mut T,
    release: Option<unsafe extern "C" fn(input: *mut T)>,
}

impl<'a, T> Opaque<T> {
    pub fn new(ptr: *mut T, release: Option<unsafe extern "C" fn(input: *mut T)>) -> Opaque<T> {
        Opaque { ptr, release }
    }

    pub(crate) fn get_ptr(&self) -> *const T {
        self.ptr
    }

    pub(crate) fn get_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<'a, T> Drop for Opaque<T> {
    fn drop(&mut self) {
        unsafe {
            self.release.map(|f| f(self.ptr));
            self.release = None;
        }
    }
}
//-------------------------------------------------------------------------------------------------
// TYPE IMPLEMENTATIONS
//-------------------------------------------------------------------------------------------------

impl Onnx {
    //---------------------------------------------------------------------------------------------
    // PUBLIC
    //---------------------------------------------------------------------------------------------
    /// Create a new instance of the ONNX API with the given LoggingLevel of the environment.
    pub fn new() -> Result<Onnx, OnnxError> {
        unsafe {
            let ort = OrtGetApiBase();
            if ort.is_null() {
                Err(OnnxError::ApiFunctionError("OrtGetApiBase is null"))
            } else {
                let get_api = try_get_fn!(*ort, GetApi);
                let api = get_api(ORT_API_VERSION);
                if api.is_null() {
                    Err(OnnxError::ApiFunctionError("Api"))
                } else {
                    let api = *api;
                    Ok(Onnx { api })
                }
            }
        }
    }

    pub fn create_session(self, model_path: &str) -> Result<Session, OnnxError> {
        let options = SessionOptions::new();
        Session::new(self.api, model_path, &options)
    }

    pub fn create_session_with_options(
        self,
        model_path: &str,
        options: &SessionOptions,
    ) -> Result<Session, OnnxError> {
        Session::new(self.api, model_path, &options)
    }
}

//---------------------------------------------------------------------------------------------
// PRIVATE FUNCTIONS
//---------------------------------------------------------------------------------------------
/// Verify that the status is ok. A status that is NULL is considered ok, otherwise it would
/// represent an error condition.
fn check_status(api: &OrtApi, status: *mut OrtStatus) -> Result<(), OnnxError> {
    if status.is_null() {
        Ok(())
    } else {
        unsafe {
            let get_error_str = try_get_fn!(api, GetErrorMessage);
            let char_ptr = get_error_str(status);
            let c_str = CStr::from_ptr(char_ptr);
            let error_message = c_str
                .to_str()
                .map(|s| s.to_string())
                .map_err(|_| OnnxError::InvalidErrorMessage)?;

            let release_status = try_get_fn!(api, ReleaseStatus);
            release_status(status);
            Err(OnnxError::Status(error_message))
        }
    }
}

//---------------------------------------------------------------------------------------------
// TESTS
//---------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    pub fn get_model_path(filename: &str) -> String {
        let mut buf: PathBuf = std::env::current_exe().unwrap();
        while buf.pop() && !buf.ends_with("onnxruntime") {}
        buf.push("csharp");
        buf.push("testdata");
        buf.push(filename);
        buf.to_str().unwrap().into()
    }
}
