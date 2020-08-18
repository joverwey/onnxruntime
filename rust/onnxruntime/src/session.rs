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
    check_status, node::Node, tensor::Tensor, tensor_element::TensorElement, LoggingLevel,
    OnnxError, Opaque, ShapedData,
};
use onnxruntime_sys::{
    GraphOptimizationLevel, ONNXTensorElementDataType, OrtAllocator, OrtApi, OrtEnv,
    OrtLoggingLevel, OrtRunOptions, OrtSession, OrtSessionOptions,
    OrtSessionOptionsAppendExecutionProvider_CUDA, OrtTensorTypeAndShapeInfo, OrtTypeInfo,
    OrtValue,
};
use std::{
    ffi::{c_void, CString},
    ptr,
};

#[cfg(target_os = "windows")]
use widestring::U16CString;

pub struct SessionOptions {
    /// The GPU Device Id if any.
    /// This is typically 0 if there is only one GPU on the system.
    /// Note that this requires the 'gpu' feature to be enabled.
    pub gpu_device_id: Option<usize>,

    /// The logging level
    pub log_level: LoggingLevel,

    /// The intra op number of threads.
    pub intra_op_num_threads: Option<usize>,

    ///The inter op number of threads.
    pub inter_op_num_threads: Option<usize>,
}

/// An ONNX session for a given model. Create tensors that match the input nodes and pass these to the *run* method.
pub struct Session {
    api: OrtApi,
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

impl SessionOptions {
    pub fn new() -> SessionOptions {
        SessionOptions {
            log_level: LoggingLevel::Warning,
            inter_op_num_threads: None,
            intra_op_num_threads: None,
            gpu_device_id: None,
        }
    }

    pub fn with_intra_op_num_treads(mut self, number_of_threads: usize) -> Self {
        self.intra_op_num_threads = Some(number_of_threads);
        self
    }

    pub fn with_inter_op_num_treads(mut self, number_of_threads: usize) -> Self {
        self.inter_op_num_threads = Some(number_of_threads);
        self
    }
}

impl<'a> Opaque<OrtSessionOptions> {
    //---------------------------------------------------------------------------------------------
    /// Create SessionOptions that allows control of the number of threads used
    /// and the kind of optimizations that are applied to the computation graph.
    pub(crate) fn from_options(
        api: &'a OrtApi,
        options: &SessionOptions,
    ) -> Result<Opaque<OrtSessionOptions>, OnnxError> {
        let mut onnx_options = try_create_opaque!(
            api,
            CreateSessionOptions,
            OrtSessionOptions,
            api.ReleaseSessionOptions
        );
        let ptr = onnx_options.get_mut_ptr();
        if let Some(inter_op_num_threads) = options.inter_op_num_threads {
            try_invoke!(api, SetInterOpNumThreads, ptr, inter_op_num_threads as i32);
        }

        if let Some(intra_op_num_threads) = options.intra_op_num_threads {
            try_invoke!(api, SetIntraOpNumThreads, ptr, intra_op_num_threads as i32);
        }

        try_invoke!(
            api,
            SetSessionGraphOptimizationLevel,
            ptr,
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        if cfg!(feature = "gpu") {
            let gpu_device_id = options.gpu_device_id.unwrap_or(0) as i32;

            invoke_fn!(
                &api,
                OrtSessionOptionsAppendExecutionProvider_CUDA,
                ptr,
                gpu_device_id
            );
        }

        Ok(onnx_options)
    }
}

#[cfg(target_os = "windows")]
fn get_model_path_from_str(model_path: &str) -> Result<U16CString, OnnxError> {
    U16CString::from_str(model_path).map_err(|_| OnnxError::InvalidString(model_path.to_string()))
}

// And this function only gets compiled if the target OS is *not* windows
#[cfg(not(target_os = "windows"))]
fn get_model_path_from_str(model_path: &str) -> Result<CString, OnnxError> {
    CString::new(model_path).map_err(|_| OnnxError::InvalidString(model_path.to_string()))
}

impl Session {
    //---------------------------------------------------------------------------------------------
    /// Create a new session for the model at the given path.
    pub(crate) fn new(
        api: OrtApi,
        model_path: &str,
        options: &SessionOptions,
    ) -> Result<Session, OnnxError> {
        let env = create_env(&api, options.log_level)?;

        let model_path = get_model_path_from_str(model_path)?;

        let options = Opaque::from_options(&api, options)?;
        let onnx_session = try_create_opaque!(
            &api,
            CreateSession,
            OrtSession,
            api.ReleaseSession,
            env.get_ptr(),
            model_path.as_ptr(),
            options.get_ptr()
        );

        let session = onnx_session.get_ptr();
        let mut input_count = 0;
        try_invoke!(&api, SessionGetInputCount, session, &mut input_count);
        let mut output_count = 0;
        try_invoke!(&api, SessionGetOutputCount, session, &mut output_count);

        let allocator = try_create!(&api, GetAllocatorWithDefaultOptions, OrtAllocator);

        let mut inputs: Vec<Node> = Vec::new();
        let mut outputs: Vec<Node> = Vec::new();

        for i in 0..input_count as usize {
            inputs.push(try_get_node_info!(
                &api,
                SessionGetInputName,
                SessionGetInputTypeInfo,
                session,
                allocator,
                i
            )?);
        }

        for i in 0..output_count as usize {
            outputs.push(try_get_node_info!(
                &api,
                SessionGetOutputName,
                SessionGetOutputTypeInfo,
                session,
                allocator,
                i
            )?);
        }

        let run_options =
            try_create_opaque!(&api, CreateRunOptions, OrtRunOptions, api.ReleaseRunOptions);

        Ok(Session {
            api,
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
            &self.api,
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
            .map(|ptr| Opaque::new(ptr, self.api.ReleaseValue))
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
            &self.api,
            CreateTensorAsOrtValue,
            OrtValue,
            self.api.ReleaseValue,
            self.allocator,
            shape_i64.as_ptr() as *const i64,
            shape.len() as u64,
            element_type
        );

        let data_ptr = try_create!(
            &self.api,
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
            &self.api,
            IsTensor,
            ptr,
            &mut is_tensor_int as *mut std::os::raw::c_int
        );

        Ok(is_tensor_int == 1)
    }

    fn get_tensor_from_value(&self, mut value: Opaque<OrtValue>) -> Result<Tensor, OnnxError> {
        if self.is_tensor(value.get_ptr())? {
            let data_ptr = try_create!(
                &self.api,
                GetTensorMutableData,
                std::os::raw::c_void,
                value.get_mut_ptr()
            );

            let shape_info = try_create_opaque!(
                &self.api,
                GetTensorTypeAndShape,
                OrtTensorTypeAndShapeInfo,
                self.api.ReleaseTensorTypeAndShapeInfo,
                value.get_mut_ptr()
            );

            let (shape, data_type) = get_shape_and_type(&self.api, shape_info.get_ptr())?;

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

fn create_env(api: &OrtApi, log_level: LoggingLevel) -> Result<Opaque<OrtEnv>, OnnxError> {
    let log_level = match log_level {
        LoggingLevel::Verbose => OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
        LoggingLevel::Info => OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
        LoggingLevel::Warning => OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        LoggingLevel::Error => OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
        LoggingLevel::Fatal => OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL,
    };

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

    use crate::{shaped_data::ShapedData, tests::get_model_path, Onnx, OnnxError};
    use assert_approx_eq::assert_approx_eq;
    use std::convert::TryInto;

    #[test]
    fn can_run_squeezenet_model() -> Result<(), OnnxError> {
        let onnx = Onnx::new()?;
        let model_path = get_model_path("squeezenet.onnx");

        let mut session = onnx.create_session(&model_path)?;

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
