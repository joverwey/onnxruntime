use crate::{check_status, get_path_from_str, OnnxError, Opaque, ONNX_NATIVE};
use onnxruntime_sys::{
    ExecutionMode, GraphOptimizationLevel, OrtApi, OrtLoggingLevel, OrtSessionOptions,
    OrtSessionOptionsAppendExecutionProvider_CUDA,
};
use std::ffi::CString;

pub struct SessionOptions {
    native_options: Opaque<OrtSessionOptions>,

    /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
    enable_memory_pattern: bool,
    enable_profiling: bool,
    is_cpu_mem_arena_enabled: bool,
    profile_output_path_prefix: String,
    log_id: String,
    execution_mode: ExecutionMode,

    /// The GPU Device Id if any.
    /// This is typically 0 if there is only one GPU on the system.
    /// Note that this requires the 'gpu' feature to be enabled.
    gpu_device_id: usize,

    /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
    /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
    log_verbosity_level: usize,

    /// The logging level
    log_severity_level: OrtLoggingLevel,

    /// Sets the number of threads used to parallelize the execution within nodes
    /// A value of None or 0 means ORT will pick a default
    intra_op_num_threads: usize,

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of None or 0 means ORT will pick a default
    inter_op_num_threads: usize,

    /// Graph optimization level
    graph_optimization_level: GraphOptimizationLevel,
}

impl SessionOptions {
    pub fn new() -> Result<SessionOptions, OnnxError> {
        let native_options = try_create_opaque!(
            &ONNX_NATIVE,
            CreateSessionOptions,
            OrtSessionOptions,
            ONNX_NATIVE.ReleaseSessionOptions
        );

        Ok(SessionOptions {
            native_options,
            log_severity_level: OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            log_verbosity_level: 0,
            inter_op_num_threads: 0,
            intra_op_num_threads: 0,
            gpu_device_id: 0,
            enable_memory_pattern: true,
            enable_profiling: false,
            is_cpu_mem_arena_enabled: true,
            profile_output_path_prefix: "onnxruntime_profile_".to_string(),
            log_id: String::new(),
            graph_optimization_level: GraphOptimizationLevel::ORT_ENABLE_ALL,
            execution_mode: ExecutionMode::ORT_SEQUENTIAL,
        })
    }

    fn get_native_ptr(&mut self) -> *mut OrtSessionOptions {
        self.native_options.get_mut_ptr()
    }

    pub fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }

    pub fn cpu_mem_arena_enabled(&self) -> bool {
        self.is_cpu_mem_arena_enabled
    }

    pub fn memory_pattern_enabled(&self) -> bool {
        self.enable_memory_pattern
    }

    pub fn profile_output_path_prefix(&self) -> &str {
        &self.profile_output_path_prefix
    }

    pub fn log_id(&self) -> &str {
        &self.log_id
    }

    pub fn profiling_enabled(&self) -> bool {
        self.enable_profiling
    }
    pub fn gpu_device_id(&self) -> usize {
        self.gpu_device_id
    }
    pub fn log_verbosity_level(&self) -> usize {
        self.log_verbosity_level
    }
    pub fn log_severity_level(&self) -> OrtLoggingLevel {
        self.log_severity_level
    }
    pub fn intra_op_num_threads(&self) -> usize {
        self.intra_op_num_threads
    }
    pub fn inter_op_num_threads(&self) -> usize {
        self.inter_op_num_threads
    }
    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel {
        self.graph_optimization_level
    }

    pub fn set_log_severity_level(
        &mut self,
        log_severity_level: OrtLoggingLevel,
    ) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetSessionLogSeverityLevel,
            self.get_native_ptr(),
            log_severity_level as i32
        );
        self.log_severity_level = log_severity_level;
        Ok(())
    }

    pub fn set_log_verbosity_level(&mut self, log_verbosity_level: usize) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetSessionLogVerbosityLevel,
            self.get_native_ptr(),
            log_verbosity_level as i32
        );
        self.log_verbosity_level = log_verbosity_level;
        Ok(())
    }

    pub fn set_intra_op_num_threads(&mut self, number_of_threads: usize) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetInterOpNumThreads,
            self.get_native_ptr(),
            number_of_threads as i32
        );
        self.intra_op_num_threads = number_of_threads;
        Ok(())
    }

    pub fn set_inter_op_num_threads(&mut self, number_of_threads: usize) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetInterOpNumThreads,
            self.get_native_ptr(),
            number_of_threads as i32
        );
        self.inter_op_num_threads = number_of_threads;
        Ok(())
    }

    pub fn set_execution_mode(&mut self, execution_mode: ExecutionMode) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetSessionExecutionMode,
            self.get_native_ptr(),
            execution_mode
        );
        self.execution_mode = execution_mode;
        Ok(())
    }

    pub fn enable_memory_pattern(&mut self) -> Result<(), OnnxError> {
        try_invoke!(&ONNX_NATIVE, EnableMemPattern, self.get_native_ptr());
        self.enable_memory_pattern = true;
        Ok(())
    }

    pub fn disable_memory_pattern(&mut self) -> Result<(), OnnxError> {
        try_invoke!(&ONNX_NATIVE, EnableMemPattern, self.get_native_ptr());
        self.enable_memory_pattern = false;
        Ok(())
    }

    pub fn enable_profiling(&mut self) -> Result<(), OnnxError> {
        let path_prefix = get_path_from_str(&self.profile_output_path_prefix)?;
        try_invoke!(
            &ONNX_NATIVE,
            EnableProfiling,
            self.get_native_ptr(),
            path_prefix.as_ptr()
        );
        self.enable_profiling = true;
        Ok(())
    }

    pub fn disable_profiling(&mut self) -> Result<(), OnnxError> {
        try_invoke!(&ONNX_NATIVE, DisableProfiling, self.get_native_ptr());
        self.enable_profiling = false;
        Ok(())
    }

    pub fn enable_cpu_mem_arena(&mut self) -> Result<(), OnnxError> {
        try_invoke!(&ONNX_NATIVE, EnableCpuMemArena, self.get_native_ptr());
        self.is_cpu_mem_arena_enabled = true;
        Ok(())
    }

    pub fn disable_cpu_mem_arena(&mut self) -> Result<(), OnnxError> {
        try_invoke!(&ONNX_NATIVE, DisableCpuMemArena, self.get_native_ptr());
        self.is_cpu_mem_arena_enabled = false;
        Ok(())
    }

    pub fn set_graph_optimization_level(
        &mut self,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> Result<(), OnnxError> {
        try_invoke!(
            &ONNX_NATIVE,
            SetSessionGraphOptimizationLevel,
            self.get_native_ptr(),
            graph_optimization_level
        );
        self.graph_optimization_level = graph_optimization_level;
        Ok(())
    }

    pub fn set_profile_output_path_prefix<S: Into<String>>(
        &mut self,
        profile_output_path_prefix: S,
    ) {
        self.profile_output_path_prefix = profile_output_path_prefix.into();
    }

    pub fn set_log_id<S: Into<String>>(&mut self, log_id: S) -> Result<(), OnnxError> {
        self.log_id = log_id.into();

        let c_str = CString::new(self.log_id.clone()).unwrap(); // This should be safe since Rust strings consist of valid UTF8 and cannot contain a \0 byte.

        try_invoke!(
            &ONNX_NATIVE,
            SetSessionLogId,
            self.get_native_ptr(),
            c_str.into_raw()
        );

        Ok(())
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

        try_invoke!(
            api,
            SetSessionGraphOptimizationLevel,
            ptr,
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        if cfg!(feature = "gpu") {
            let gpu_device_id = options.gpu_device_id as i32;

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

//==================================================================================================
// TESTS
//==================================================================================================

#[cfg(test)]
mod tests {
    use super::{ExecutionMode, GraphOptimizationLevel, OnnxError, OrtLoggingLevel};
    use crate::session_options::SessionOptions;
    #[test]
    fn test_session_options() -> Result<(), OnnxError> {
        let mut opt = SessionOptions::new()?;

        // check default values of the properties
        assert_eq!(ExecutionMode::ORT_SEQUENTIAL, opt.execution_mode());
        assert!(opt.memory_pattern_enabled());
        assert!(!opt.profiling_enabled());
        assert_eq!("onnxruntime_profile_", opt.profile_output_path_prefix());
        assert!(opt.cpu_mem_arena_enabled());
        assert_eq!("", opt.log_id());
        assert_eq!(0, opt.log_verbosity_level());
        assert_eq!(
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            opt.log_severity_level()
        );
        assert_eq!(0, opt.intra_op_num_threads);
        assert_eq!(0, opt.inter_op_num_threads());
        assert_eq!(
            GraphOptimizationLevel::ORT_ENABLE_ALL,
            opt.graph_optimization_level()
        );

        // try setting options
        opt.set_execution_mode(ExecutionMode::ORT_PARALLEL)?;
        opt.disable_memory_pattern()?;
        opt.enable_profiling()?;
        opt.set_profile_output_path_prefix("Ort_P_");
        opt.disable_cpu_mem_arena()?;
        opt.set_log_id("MyLogId")?;
        opt.set_log_verbosity_level(1)?;
        opt.set_log_severity_level(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)?;
        opt.set_intra_op_num_threads(4)?;
        opt.set_inter_op_num_threads(4)?;
        opt.set_graph_optimization_level(GraphOptimizationLevel::ORT_ENABLE_EXTENDED)?;

        assert_eq!(ExecutionMode::ORT_PARALLEL, opt.execution_mode());
        assert!(!opt.memory_pattern_enabled());
        assert!(opt.profiling_enabled());
        assert_eq!("Ort_P_", opt.profile_output_path_prefix());
        assert!(!opt.cpu_mem_arena_enabled());
        assert_eq!("MyLogId", opt.log_id());
        assert_eq!(1, opt.log_verbosity_level());
        assert_eq!(
            OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
            opt.log_severity_level()
        );
        assert_eq!(4, opt.intra_op_num_threads());
        assert_eq!(4, opt.inter_op_num_threads());
        assert_eq!(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
            opt.graph_optimization_level()
        );

        // Assert.Throws<OnnxRuntimeException>(() => { opt.GraphOptimizationLevel = (GraphOptimizationLevel)10; });
        //
        // opt.AppendExecutionProvider_CPU(1);
        // #if USE_DNNL
        // opt.AppendExecutionProvider_Dnnl(0);
        // #endif
        // #if USE_CUDA
        // opt.AppendExecutionProvider_CUDA(0);
        // #endif
        // #if USE_NGRAPH
        // opt.AppendExecutionProvider_NGraph("CPU");  //TODO: this API should be refined
        // #endif
        // #if USE_OPENVINO
        // opt.AppendExecutionProvider_OpenVINO();
        // #endif
        // #if USE_TENSORRT
        // opt.AppendExecutionProvider_Tensorrt(0);
        // #endif
        // #if USE_MIGRAPHX
        // opt.AppendExecutionProvider_MIGraphX(0);
        // #endif
        // #if USE_NNAPI
        // opt.AppendExecutionProvider_Nnapi();
        // #endif

        Ok(())
    }
}
