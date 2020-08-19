use crate::{check_status, OnnxError, OnnxObject, ONNX_NATIVE};
use onnxruntime_sys::{OrtLoggingLevel, OrtRunOptions};
use std::ffi::{CStr, CString};
use std::ptr;

/// Configuration information for a single Run.
pub struct RunOptions {
    native_options: OnnxObject<OrtRunOptions>,

    /// Log Verbosity Level for a particular Run() invocation. Default = 0. Valid values are >=0.
    /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
    log_verbosity_level: usize,
    /// Log Severity Level for a particular Run() invocation. Default = ORT_LOGGING_LEVEL_WARNING
    log_severity_level: OrtLoggingLevel,

    log_id: Option<CString>,
}

impl RunOptions {

    /// Create a new instance of default RunOptions
    pub fn new() -> Result<RunOptions, OnnxError> {
        let native_options = try_create_opaque!(
            CreateRunOptions,
            OrtRunOptions,
            ReleaseRunOptions
        );

        Ok(RunOptions {
            native_options,
            log_severity_level: OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            log_verbosity_level: 0,
            log_id: None
        })
    }
    /// Log Verbosity Level for the run. Default = 0. Valid values are >=0.
    pub fn log_verbosity_level(&self) -> usize {
        self.log_verbosity_level
    }

    /// Log Severity Level for the run. Default = ORT_LOGGING_LEVEL_WARNING
    pub fn log_severity_level(&self) -> OrtLoggingLevel {
        self.log_severity_level
    }

    /// Log Id to be used for the run.
    pub fn log_id(&self) -> Result<String, OnnxError> {
            let tag_ptr: *mut *const ::std::os::raw::c_char = ptr::null_mut();
            try_invoke!(RunOptionsGetRunTag, self.native_options.get_ptr(), tag_ptr);

            let tag = unsafe { CStr::from_ptr(*tag_ptr) };

            let log_id = tag.to_str().expect("We expect ONNX to return valid UTF8");

            Ok(log_id.to_string())
    }

    /// Set Log Severity Level for a particular Run() invocation. Default = ORT_LOGGING_LEVEL_WARNING
    pub fn set_log_severity_level(
        &mut self,
        log_severity_level: OrtLoggingLevel,
    ) -> Result<(), OnnxError> {
        try_invoke!(
            RunOptionsSetRunLogSeverityLevel,
            self.native_options.get_mut_ptr(),
            log_severity_level as i32
        );
        self.log_severity_level = log_severity_level;
        Ok(())
    }

    /// Sets the Log Verbosity Level for a particular Run() invocation. Default = 0. Valid values are >=0.
    /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
    pub fn set_log_verbosity_level(&mut self, log_verbosity_level: usize) -> Result<(), OnnxError> {
        try_invoke!(
            RunOptionsSetRunLogVerbosityLevel,
            self.native_options.get_mut_ptr(),
            log_verbosity_level as i32
        );
        self.log_verbosity_level = log_verbosity_level;
        Ok(())
    }

    /// Sets the Log Id to be used for a particular Run() invocation. Default is empty string.
    pub fn set_log_id(&mut self, log_id: &str) -> Result<(), OnnxError> {
        let c_string = CString::new(log_id).unwrap(); // This should be safe since Rust strings consist of valid UTF8 and cannot contain a \0 byte.

        try_invoke!(RunOptionsSetRunTag, self.native_options.get_mut_ptr(), c_string.as_ptr());

        // ONNX just keeps a reference to the pointer so we need to store the text so that it
        // outlives the stack frame.
        self.log_id = Some(c_string);
        Ok(())
    }
}