use crate::{tensor_element::TensorElement, OnnxError, OnnxObject};
use onnxruntime_sys::{ONNXTensorElementDataType, OrtValue};
use std::{ffi::c_void, ptr};

/// An ONNX Tensor
pub struct Tensor {
    data_ptr: *mut c_void,
    value: OnnxObject<OrtValue>,
    pub(crate) element_type: ONNXTensorElementDataType,
    shape: Vec<usize>,
}

impl Tensor {
    pub(crate) fn new(
        data_ptr: *mut c_void,
        value: OnnxObject<OrtValue>,
        element_type: ONNXTensorElementDataType,
        shape: Vec<usize>,
    ) -> Tensor {
        Tensor {
            data_ptr,
            value,
            element_type,
            shape,
        }
    }

    /// The total number of elements in this tensor when flattened.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// The number of elements in each dimension.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn value(&self) -> &OnnxObject<OrtValue> {
        &self.value
    }

    pub(crate) fn copy_data<T: TensorElement>(&self) -> Result<Vec<T>, OnnxError> {
        if self.element_type != T::get_type() {
            return Err(OnnxError::TensorTypeMismatch(
                format!("{}", T::get_type()),
                format!("{}", self.element_type),
            ));
        }
        let void_ptr = self.data_ptr;
        let count = self.element_count();
        let mut data: Vec<T> = Vec::with_capacity(count);
        unsafe {
            data.set_len(count);
            ptr::copy(void_ptr as *mut T, data.as_mut_ptr(), count);
        };

        Ok(data)
    }
}
