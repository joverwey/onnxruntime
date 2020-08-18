use crate::{check_status, get_path_from_str, OnnxError, Opaque, ONNX_NATIVE};
use onnxruntime_sys::{
    ExecutionMode, GraphOptimizationLevel, OrtLoggingLevel, OrtSessionOptions,
    OrtSessionOptionsAppendExecutionProvider_CPU, OrtSessionOptionsAppendExecutionProvider_CUDA,
};
use std::ffi::CString;

#[cfg(target_os = "windows")]
use libloading::Library;

/// Holds the options for creating an InferenceSession
pub struct SessionOptions {
    native_options: Opaque<OrtSessionOptions>,

    /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
    enable_memory_pattern: bool,
    enable_profiling: bool,
    is_cpu_mem_arena_enabled: bool,
    profile_output_path_prefix: String,
    log_id: String,
    execution_mode: ExecutionMode,

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
    /// Constructs an empty SessionOptions with default options
    pub fn new() -> Result<SessionOptions, OnnxError> {
        let native_options = try_create_opaque!(
            CreateSessionOptions,
            OrtSessionOptions,
            ReleaseSessionOptions
        );

        Ok(SessionOptions {
            native_options,
            log_severity_level: OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            log_verbosity_level: 0,
            inter_op_num_threads: 0,
            intra_op_num_threads: 0,
            enable_memory_pattern: true,
            enable_profiling: false,
            is_cpu_mem_arena_enabled: true,
            profile_output_path_prefix: "onnxruntime_profile_".to_string(),
            log_id: String::new(),
            graph_optimization_level: GraphOptimizationLevel::ORT_ENABLE_ALL,
            execution_mode: ExecutionMode::ORT_SEQUENTIAL,
        })
    }

    /// A helper method to construct a SessionOptions object for CUDA execution.
    /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
    /// Returns a SessionsOptions() object configured for execution on deviceId or an error.
    pub fn with_cuda(device_id: usize) -> Result<SessionOptions, OnnxError> {
        let mut options = SessionOptions::new()?;

        options.append_execution_provider_cuda(device_id)?;

        invoke_fn!(
            OrtSessionOptionsAppendExecutionProvider_CPU,
            options.get_ptr_mut(),
            1
        );
        Ok(options)
    }

    /// A helper method to construct a SessionOptions object for CUDA execution.
    /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
    /// Returns a SessionsOptions() object configured for execution on deviceId or an error.
    pub fn with_cuda_deafult() -> Result<SessionOptions, OnnxError> {
        Self::with_cuda(0)
    }

    pub(crate) fn get_ptr(&self) -> *const OrtSessionOptions {
        self.native_options.get_ptr()
    }

    pub(crate) fn get_ptr_mut(&mut self) -> *mut OrtSessionOptions {
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
            SetSessionLogSeverityLevel,
            self.get_ptr_mut(),
            log_severity_level as i32
        );
        self.log_severity_level = log_severity_level;
        Ok(())
    }

    pub fn set_log_verbosity_level(&mut self, log_verbosity_level: usize) -> Result<(), OnnxError> {
        try_invoke!(
            SetSessionLogVerbosityLevel,
            self.get_ptr_mut(),
            log_verbosity_level as i32
        );
        self.log_verbosity_level = log_verbosity_level;
        Ok(())
    }

    pub fn set_intra_op_num_threads(&mut self, number_of_threads: usize) -> Result<(), OnnxError> {
        try_invoke!(
            SetInterOpNumThreads,
            self.get_ptr_mut(),
            number_of_threads as i32
        );
        self.intra_op_num_threads = number_of_threads;
        Ok(())
    }

    pub fn set_inter_op_num_threads(&mut self, number_of_threads: usize) -> Result<(), OnnxError> {
        try_invoke!(
            SetInterOpNumThreads,
            self.get_ptr_mut(),
            number_of_threads as i32
        );
        self.inter_op_num_threads = number_of_threads;
        Ok(())
    }

    pub fn set_execution_mode(&mut self, execution_mode: ExecutionMode) -> Result<(), OnnxError> {
        try_invoke!(SetSessionExecutionMode, self.get_ptr_mut(), execution_mode);
        self.execution_mode = execution_mode;
        Ok(())
    }

    pub fn enable_memory_pattern(&mut self) -> Result<(), OnnxError> {
        try_invoke!(EnableMemPattern, self.get_ptr_mut());
        self.enable_memory_pattern = true;
        Ok(())
    }

    pub fn disable_memory_pattern(&mut self) -> Result<(), OnnxError> {
        try_invoke!(EnableMemPattern, self.get_ptr_mut());
        self.enable_memory_pattern = false;
        Ok(())
    }

    pub fn enable_profiling(&mut self) -> Result<(), OnnxError> {
        let path_prefix = get_path_from_str(&self.profile_output_path_prefix)?;
        try_invoke!(EnableProfiling, self.get_ptr_mut(), path_prefix.as_ptr());
        self.enable_profiling = true;
        Ok(())
    }

    pub fn disable_profiling(&mut self) -> Result<(), OnnxError> {
        try_invoke!(DisableProfiling, self.get_ptr_mut());
        self.enable_profiling = false;
        Ok(())
    }

    pub fn enable_cpu_mem_arena(&mut self) -> Result<(), OnnxError> {
        try_invoke!(EnableCpuMemArena, self.get_ptr_mut());
        self.is_cpu_mem_arena_enabled = true;
        Ok(())
    }

    pub fn disable_cpu_mem_arena(&mut self) -> Result<(), OnnxError> {
        try_invoke!(DisableCpuMemArena, self.get_ptr_mut());
        self.is_cpu_mem_arena_enabled = false;
        Ok(())
    }

    pub fn set_graph_optimization_level(
        &mut self,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> Result<(), OnnxError> {
        try_invoke!(
            SetSessionGraphOptimizationLevel,
            self.get_ptr_mut(),
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

    /// Use only if you have the onnxruntime package specific to this Execution Provider.
    pub fn append_execution_provider_cuda(&mut self, device_id: usize) -> Result<(), OnnxError> {
        if cfg!(feature = "gpu") {
            check_cuda_execution_provider_dlls()?;
            invoke_fn!(
                OrtSessionOptionsAppendExecutionProvider_CUDA,
                self.get_ptr_mut(),
                device_id as i32
            );
        } else {
            return Err(OnnxError::CudaNotSupported);
        }

        Ok(())
    }

    pub fn append_execution_provider_cpu(&mut self, use_arena: usize) -> Result<(), OnnxError> {
        invoke_fn!(
            OrtSessionOptionsAppendExecutionProvider_CPU,
            self.get_ptr_mut(),
            use_arena as i32
        );
        Ok(())
    }

    pub fn set_log_id<S: Into<String>>(&mut self, log_id: S) -> Result<(), OnnxError> {
        self.log_id = log_id.into();

        let c_str = CString::new(self.log_id.clone()).unwrap(); // This should be safe since Rust strings consist of valid UTF8 and cannot contain a \0 byte.

        try_invoke!(SetSessionLogId, self.get_ptr_mut(), c_str.into_raw());

        Ok(())
    }
}

static CUDA_DELAY_LOADED_LIBS: &'static [&'static str] =
    &["cublas64_10.dll", "cudnn64_7.dll", "curand64_10.dll"];

fn check_cuda_execution_provider_dlls() -> Result<(), OnnxError> {
    if cfg!(target_os = "windows") {
        for dll in CUDA_DELAY_LOADED_LIBS {
            match Library::new(dll) {
                Ok(_) => (),
                Err(_) => return Err(OnnxError::CudaNotFound),
            };
        }
    }

    Ok(())
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

        opt.append_execution_provider_cpu(1)?;
        if cfg!(feature = "gpu") {
            opt.append_execution_provider_cuda(0)?;
        }

        Ok(())
    }
}
