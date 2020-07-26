use onnxruntime_sys::ONNXTensorElementDataType;

pub trait TensorElement {
    fn get_type() -> ONNXTensorElementDataType;
}

impl TensorElement for f64 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    }
}

impl TensorElement for i64 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    }
}

impl TensorElement for u64 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    }
}

impl TensorElement for u32 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    }
}

impl TensorElement for i32 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    }
}

impl TensorElement for f32 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }
}

impl TensorElement for u16 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    }
}

impl TensorElement for i16 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    }
}

impl TensorElement for u8 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    }
}

impl TensorElement for i8 {
    fn get_type() -> ONNXTensorElementDataType {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    }
}
