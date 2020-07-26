use crate::{tensor_element::TensorElement, OnnxError, Opaque};
use onnxruntime_sys::{ONNXTensorElementDataType, OrtValue};
use std::{ffi::c_void, ptr};
pub struct Tensor {
    data_ptr: *mut c_void,
    value: Opaque<OrtValue>,
    pub(crate) element_type: ONNXTensorElementDataType,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(
        data_ptr: *mut c_void,
        value: Opaque<OrtValue>,
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

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn value(&self) -> &Opaque<OrtValue> {
        &self.value
    }

    pub fn copy_data<T: TensorElement>(&self) -> Result<Vec<T>, OnnxError> {
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
