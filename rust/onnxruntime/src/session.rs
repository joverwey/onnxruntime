macro_rules! try_get_node_info {
    ($api: expr, $get_name: ident, $get_type_info: ident, $session: expr, $allocator: expr, $index: expr) => {{
        let mut raw_name: *mut ::std::os::raw::c_char = std::ptr::null_mut();
        try_invoke!(
            $api,
            $get_name,
            $session,
            $index as u64,
            $allocator,
            &mut raw_name
        );

        let type_info = try_create!($api, $get_type_info, OrtTypeInfo, $session, $index as u64);

        try_get_node($api, type_info, raw_name, $allocator)
    }};
}
use crate::{
    check_status, get_path_from_str, node::Node, tensor::Tensor, tensor_element::TensorElement,
    OnnxError, Opaque, ShapedData, ONNX_NATIVE,
};
use onnxruntime_sys::{
    ONNXTensorElementDataType, OrtAllocator, OrtApi, OrtEnv, OrtLoggingLevel, OrtRunOptions,
    OrtSession, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue,
};
use std::{
    ffi::{c_void, CString},
    ptr,
};

use crate::session_options::SessionOptions;

/// An ONNX session for a given model. Create tensors that match the input nodes and pass these to the *run* method.
pub struct Session {
    // The environment has to stay alive for the duration of the session as it is used for internal logging.
    #[allow(dead_code)]
    env: Opaque<OrtEnv>,
    allocator: *mut OrtAllocator,
    /// The input nodes.
    inputs: Vec<Node>,

    /// The output nodes.
    outputs: Vec<Node>,
    onnx_session: Opaque<OrtSession>,
    run_options: Opaque<OrtRunOptions>,
}

impl Session {
    pub fn new(model_path: &str) -> Result<Session, OnnxError> {
        let options = SessionOptions::new()?;
        Self::from_options(model_path, &options)
    }
    //---------------------------------------------------------------------------------------------
    /// Create a new session for the model at the given path.
    pub(crate) fn from_options(
        model_path: &str,
        options: &SessionOptions,
    ) -> Result<Session, OnnxError> {
        let env = create_env(&ONNX_NATIVE, options.log_severity_level())?;

        let model_path = get_path_from_str(model_path)?;

        let options = Opaque::from_options(&ONNX_NATIVE, options)?;
        let onnx_session = try_create_opaque!(
            &ONNX_NATIVE,
            CreateSession,
            OrtSession,
            ONNX_NATIVE.ReleaseSession,
            env.get_ptr(),
            model_path.as_ptr(),
            options.get_ptr()
        );

        let session = onnx_session.get_ptr();
        let mut input_count = 0;
        try_invoke!(
            &ONNX_NATIVE,
            SessionGetInputCount,
            session,
            &mut input_count
        );
        let mut output_count = 0;
        try_invoke!(
            &ONNX_NATIVE,
            SessionGetOutputCount,
            session,
            &mut output_count
        );

        let allocator = try_create!(&ONNX_NATIVE, GetAllocatorWithDefaultOptions, OrtAllocator);

        let mut inputs: Vec<Node> = Vec::new();
        let mut outputs: Vec<Node> = Vec::new();

        for i in 0..input_count as usize {
            inputs.push(try_get_node_info!(
                &ONNX_NATIVE,
                SessionGetInputName,
                SessionGetInputTypeInfo,
                session,
                allocator,
                i
            )?);
        }

        for i in 0..output_count as usize {
            outputs.push(try_get_node_info!(
                &ONNX_NATIVE,
                SessionGetOutputName,
                SessionGetOutputTypeInfo,
                session,
                allocator,
                i
            )?);
        }

        let run_options = try_create_opaque!(
            &ONNX_NATIVE,
            CreateRunOptions,
            OrtRunOptions,
            ONNX_NATIVE.ReleaseRunOptions
        );

        Ok(Session {
            env,
            inputs,
            allocator,
            outputs,
            onnx_session,
            run_options,
        })
    }

    /// Run the model with the given input tensors.
    /// The output tensors are returned.
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, OnnxError> {
        let input_names_c: Vec<*const i8> = self
            .inputs
            .iter()
            .map(|n| n.raw_name as *const i8)
            .collect();
        let input_count = self.inputs.len();
        let input_pointers: Vec<*const OrtValue> =
            inputs.iter().map(|input| input.value().get_ptr()).collect();

        let output_names_c: Vec<*const i8> = self
            .outputs
            .iter()
            .map(|n| n.raw_name as *const i8)
            .collect();
        let output_count = self.outputs.len();
        let mut output_pointers: Vec<*mut OrtValue> = vec![std::ptr::null_mut(); output_count];

        try_invoke!(
            &ONNX_NATIVE,
            Run,
            self.onnx_session.get_mut_ptr(),
            self.run_options.get_ptr(),
            input_names_c.as_ptr() as *const *const ::std::os::raw::c_char,
            input_pointers.as_ptr() as *const *const OrtValue,
            input_count as u64,
            output_names_c.as_ptr() as *const *const ::std::os::raw::c_char,
            output_count as u64,
            output_pointers.as_mut_ptr() as *mut *mut OrtValue
        );

        // First wrap values so we are sure they will be dropped if anything goes wrong below.
        let values: Vec<Opaque<OrtValue>> = output_pointers
            .into_iter()
            .map(|ptr| Opaque::new(ptr, ONNX_NATIVE.ReleaseValue))
            .collect();

        let x = values
            .into_iter()
            .map(|v| self.get_tensor_from_value(v))
            .collect();
        x
    }

    /// Create a tensor from the flattened data with the given shape.
    pub fn create_tensor_from_shaped_data<T: TensorElement>(
        &self,
        shaped_data: ShapedData<T>,
    ) -> Result<Tensor, OnnxError> {
        let (shape, data) = (shaped_data.shape(), shaped_data.elements());

        let element_type = T::get_type();
        let shape_i64: Vec<i64> = shape.iter().map(|&v| v as i64).collect();
        let mut value = try_create_opaque!(
            &ONNX_NATIVE,
            CreateTensorAsOrtValue,
            OrtValue,
            ONNX_NATIVE.ReleaseValue,
            self.allocator,
            shape_i64.as_ptr() as *const i64,
            shape.len() as u64,
            element_type
        );

        let data_ptr = try_create!(
            &ONNX_NATIVE,
            GetTensorMutableData,
            std::os::raw::c_void,
            value.get_mut_ptr()
        );

        unsafe { ptr::copy(data.as_ptr(), data_ptr as *mut T, data.len()) };

        Ok(Tensor::new(data_ptr, value, element_type, shape.to_vec()))
    }

    pub fn inputs(&self) -> &[Node] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Node] {
        &self.outputs
    }

    fn is_tensor(&self, ptr: *const OrtValue) -> Result<bool, OnnxError> {
        let mut is_tensor_int = 0;
        try_invoke!(
            &ONNX_NATIVE,
            IsTensor,
            ptr,
            &mut is_tensor_int as *mut std::os::raw::c_int
        );

        Ok(is_tensor_int == 1)
    }

    fn get_tensor_from_value(&self, mut value: Opaque<OrtValue>) -> Result<Tensor, OnnxError> {
        if self.is_tensor(value.get_ptr())? {
            let data_ptr = try_create!(
                &ONNX_NATIVE,
                GetTensorMutableData,
                std::os::raw::c_void,
                value.get_mut_ptr()
            );

            let shape_info = try_create_opaque!(
                &ONNX_NATIVE,
                GetTensorTypeAndShape,
                OrtTensorTypeAndShapeInfo,
                ONNX_NATIVE.ReleaseTensorTypeAndShapeInfo,
                value.get_mut_ptr()
            );

            let (shape, data_type) = get_shape_and_type(&ONNX_NATIVE, shape_info.get_ptr())?;

            if shape.iter().any(|&s| s <= 0) {
                return Err(OnnxError::InvalidTensorShape(shape));
            }

            Ok(Tensor::new(
                data_ptr,
                value,
                data_type,
                shape.iter().map(|&i| i as usize).collect(),
            ))
        } else {
            Err(OnnxError::NotATensor)
        }
    }
}

fn try_get_node(
    api: &OrtApi,
    type_info: *mut OrtTypeInfo,
    raw_name: *mut i8,
    allocator: *mut OrtAllocator,
) -> Result<Node, OnnxError> {
    let mut shape_info: *const OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
    try_invoke!(api, CastTypeInfoToTensorInfo, type_info, &mut shape_info);

    let (shape, data_type) = get_shape_and_type(api, shape_info)?;

    let _to_drop = Opaque {
        ptr: type_info,
        release: api.ReleaseTypeInfo,
    };

    Ok(Node::new(
        data_type,
        raw_name as *mut c_void,
        shape,
        allocator,
        api.AllocatorFree,
    ))
}

fn get_shape_and_type(
    api: &OrtApi,
    shape_info: *const OrtTensorTypeAndShapeInfo,
) -> Result<(Vec<i64>, ONNXTensorElementDataType), OnnxError> {
    let mut data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    try_invoke!(api, GetTensorElementType, shape_info, &mut data_type);

    let mut num_dimensions = 0;
    try_invoke!(api, GetDimensionsCount, shape_info, &mut num_dimensions);

    let mut shape = vec![0i64; num_dimensions as usize];
    try_invoke!(
        &api,
        GetDimensions,
        shape_info,
        shape.as_mut_ptr() as *mut i64,
        num_dimensions
    );

    Ok((shape, data_type))
}

fn create_env(api: &OrtApi, log_level: OrtLoggingLevel) -> Result<Opaque<OrtEnv>, OnnxError> {
    let c_str = CString::new("onnx_runtime").unwrap(); // This should be safe since Rust strings consist of valid UTF8 and cannot contain a \0 byte.
    let env = try_create_opaque!(
        api,
        CreateEnv,
        OrtEnv,
        api.ReleaseEnv,
        log_level,
        c_str.as_ptr()
    );
    Ok(env)
}

#[cfg(test)]
mod tests {

    use crate::session::Session;
    use crate::{shaped_data::ShapedData, tests::get_model_path, OnnxError};
    use assert_approx_eq::assert_approx_eq;
    use std::convert::TryInto;

    #[test]
    fn can_run_squeezenet_model() -> Result<(), OnnxError> {
        let model_path = get_model_path("squeezenet.onnx");

        let mut session = Session::new(&model_path)?;

        let shape = vec![1, 3, 224, 224];
        let input_tensor_size = shape.iter().product();

        let input_tensor_values: Vec<f32> = (0..input_tensor_size)
            .map(|i| i as f32 / (input_tensor_size as f32 + 1.0))
            .collect();

        let shaped_data = ShapedData::new(shape, input_tensor_values)?;

        let inputs = vec![session.create_tensor_from_shaped_data(shaped_data)?];

        let outputs = session.run(&inputs)?;

        assert_eq!(session.outputs.len(), 1);

        let output = &outputs[0];

        let shaped_data: ShapedData<f32> = output.try_into()?;
        let top_5: Vec<f32> = shaped_data.elements().iter().cloned().take(5).collect();

        assert_approx_eq!(top_5[0], 0.000045440636);
        assert_approx_eq!(top_5[1], 0.0038458568);
        assert_approx_eq!(top_5[2], 0.00012494661);
        assert_approx_eq!(top_5[3], 0.0011804522);
        assert_approx_eq!(top_5[4], 0.001316936);

        assert_eq!(session.inputs.len(), 1);
        assert_eq!(session.inputs[0].name(), "data_0");
        assert_eq!(session.inputs[0].shape(), &[1, 3, 224, 224]);

        assert_eq!(session.outputs.len(), 1);
        assert_eq!(session.outputs[0].name(), "softmaxout_1");
        assert!(session.outputs[0].is_a::<f32>());
        assert_eq!(session.outputs[0].shape(), &[1, 1000, 1, 1]);
        Ok(())
    }
}
