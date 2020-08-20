use crate::tensor_element::TensorElement;
use ffi::CStr;
use onnxruntime_sys::{ONNXTensorElementDataType, OrtAllocator, OrtStatus};
use std::ffi;
pub struct Node {
    pub data_type: ONNXTensorElementDataType,
    pub(crate) raw_name: *mut ffi::c_void,
    shape: Vec<i64>,
    allocator: *mut OrtAllocator,
    release: Option<
        unsafe extern "C" fn(
            ptr: *mut OrtAllocator,
            p: *mut ::std::os::raw::c_void,
        ) -> *mut OrtStatus,
    >,
}

impl Node {
    pub(crate) fn new(
        data_type: ONNXTensorElementDataType,
        raw_name: *mut ffi::c_void,
        shape: Vec<i64>,
        allocator: *mut OrtAllocator,
        release: Option<
            unsafe extern "C" fn(
                ptr: *mut OrtAllocator,
                p: *mut ::std::os::raw::c_void,
            ) -> *mut OrtStatus,
        >,
    ) -> Node {
        Node {
            data_type,
            raw_name,
            shape,
            allocator,
            release,
        }
    }

    pub fn name(&self) -> &str {
        let c_str: &CStr = unsafe { CStr::from_ptr(self.raw_name as *const i8) };
        c_str.to_str().unwrap_or("invalid_utf8_name")
    }

    pub fn data_type(&self) -> ONNXTensorElementDataType {
        self.data_type
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.shape
    }

    pub fn is_a<T: TensorElement>(&self) -> bool {
        T::get_type() == self.data_type
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        unsafe {
            self.release.map(|f| f(self.allocator, self.raw_name));
            self.release = None;
        }
    }
}
