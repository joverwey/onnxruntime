//! Example usage:
//!
//! ``` no_run
//!# use onnxruntime::shaped_data::ShapedData;
//!# use onnxruntime::session::Session;
//!# use std::path::PathBuf;
//!# use std::convert::TryInto;
//!# pub fn get_model_path(filename: &str) -> String {
//!#    let mut buf: PathBuf = std::env::current_exe().unwrap();
//!#    while buf.pop() && !buf.ends_with("target") {}
//!#    buf.push("data");
//!#    buf.push(filename);
//!#    buf.to_str().unwrap().into()
//!# }
//!let model_path = get_model_path("squeezenet.onnx");

//! //We first create a session for a given model.
//!let mut session = Session::new(&model_path).unwrap();
//!
//! //Next we need to create input tensors. The caller of the model typically knows the shape
//! //of this data but it can be verified by inspecting the *inputs()* of *session*.
//!let shape = vec![1, 3, 224, 224];
//!
//! //We populate an input vector with arbitrary values that would in practice be read from an image.
//!let input_tensor_size = shape.iter().product();
//!let input_tensor_values: Vec<f32> =
//!     (0..input_tensor_size)
//!         .map(|i| i as f32 / (input_tensor_size as f32 + 1.0))
//!         .collect();
//!
//! //We then create a tensor from the data with the given shape and run the model,
//! //in this case with a single input tensor.
//!let shaped_data = ShapedData::new(shape, input_tensor_values).unwrap();
//!let inputs = vec![session.create_tensor_from_shaped_data(shaped_data).unwrap()];
//!let outputs = session.run(&inputs).unwrap();
//!
//! //Finally convert the output tensor into a shaped data representation.
//!let output = &outputs[0];
//!let shaped_data: ShapedData<f32> = output.try_into().unwrap();
//! //For SqueezeNet, this represents a one-hot vector representing the probability of
//! //being one of 1000 classes.
//! ```
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
    ($api:expr, $function_name:ident) => {
        let status = unsafe {$function_name()};
        check_status($api, status)?
    };
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

macro_rules! try_create_opaque {
    ($api:expr, $function_name:ident, $type_name:ty, $release:expr) => {

        {
            let p = try_create!($api, $function_name, $type_name);
            Opaque::new(p, $release)
        }

    };
    ($api:expr, $function_name:ident, $type_name:ty, $release:expr $(, $param:expr)*) => {

        {
            let p = try_create!($api, $function_name, $type_name, $($param),*);
            Opaque::new(p, $release)
        }

    };
}

//-------------------------------------------------------------------------------------------------
// IMPORTS
//-------------------------------------------------------------------------------------------------
pub mod node;
pub mod session;
pub mod session_options;
pub mod shaped_data;
pub mod tensor;
pub mod tensor_element;

use onnxruntime_sys::{OrtApi, OrtGetApiBase, OrtStatus};

use lazy_static::lazy_static;
use shaped_data::ShapedData;
use std::ffi::CStr;
use thiserror::Error;
#[cfg(target_os = "windows")]
use widestring::U16CString;

//-------------------------------------------------------------------------------------------------
// CONSTANTS
//-------------------------------------------------------------------------------------------------
const ORT_API_VERSION: u32 = 2;

//-------------------------------------------------------------------------------------------------
// TYPES
//-------------------------------------------------------------------------------------------------
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

lazy_static! {
    pub static ref ONNX_NATIVE: OrtApi = {
        unsafe {
            let ort = OrtGetApiBase();
            if ort.is_null() {
                panic!("OrtGetApiBase is null");
            } else {
                let get_api = (*ort)
                    .GetApi
                    .expect("Missing GetApi function. Invalid DLL.");
                let api = get_api(ORT_API_VERSION);
                if api.is_null() {
                    panic!("Incorrect API version. Are you using the correct DLL?");
                } else {
                    *api
                }
            }
        }
    };
}

struct Opaque<T> {
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

#[cfg(target_os = "windows")]
fn get_path_from_str(model_path: &str) -> Result<U16CString, OnnxError> {
    U16CString::from_str(model_path).map_err(|_| OnnxError::InvalidString(model_path.to_string()))
}

// And this function only gets compiled if the target OS is *not* windows
#[cfg(not(target_os = "windows"))]
fn get_path_from_str(model_path: &str) -> Result<CString, OnnxError> {
    CString::new(model_path).map_err(|_| OnnxError::InvalidString(model_path.to_string()))
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
pub(crate) mod tests {
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
