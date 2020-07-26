use crate::{session::Session, tensor::Tensor, tensor_element::TensorElement, OnnxError};
use std::convert::TryFrom;

pub struct ShapedData<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T: TensorElement> ShapedData<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<ShapedData<T>, OnnxError> {
        let element_count = shape.iter().product();
        if element_count != data.len() {
            Err(OnnxError::TensorDimensionElementCountMismatch(
                format!("{:?}", shape),
                element_count,
            ))
        } else {
            Ok(ShapedData { shape, data })
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn elements(&self) -> &[T] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn into_tensor(self, session: &Session) -> Result<Tensor, OnnxError> {
        session.create_tensor_from_shaped_data(self)
    }
}

impl<'a, T: TensorElement> TryFrom<&Tensor> for ShapedData<T> {
    type Error = OnnxError;

    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        Ok(ShapedData {
            shape: tensor.shape().to_vec(),
            data: tensor.copy_data::<T>()?,
        })
    }
}
