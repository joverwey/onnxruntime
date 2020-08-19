use crate::{check_status, OnnxError, OnnxObject, ONNX_NATIVE};
use onnxruntime_sys::{OrtLoggingLevel, OrtRunOptions};
use std::ffi::{CStr, CString};

/// Configuration information for a single Run.
pub struct RunOptions {

    native_options: OnnxObject<OrtRunOptions>,

    terminate: bool,

    /// Log Verbosity Level for a particular Run() invocation. Default = 0. Valid values are >=0.
    /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
    log_verbosity_level: usize,
    /// Log Severity Level for a particular Run() invocation. Default = ORT_LOGGING_LEVEL_WARNING
    log_severity_level: OrtLoggingLevel,
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
            terminate: false
        })
    }

    pub fn should_terminate(&self) -> bool {
        self.terminate
    }

    pub fn set_terminate(&mut self, should_terminate: bool) -> Result<(), OnnxError> {
        if should_terminate != self.terminate {
            if should_terminate {
                try_invoke!(RunOptionsSetTerminate, self.native_options.get_mut_ptr());
            }
            else {
                try_invoke!(RunOptionsUnsetTerminate, self.native_options.get_mut_ptr());
            }

            self.terminate = should_terminate;
        }

        Ok(())
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
            let mut tag_ptr: *const ::std::os::raw::c_char = std::ptr::null();
            try_invoke!(RunOptionsGetRunTag, self.native_options.get_ptr(), &mut tag_ptr);

            let tag = unsafe { CStr::from_ptr(tag_ptr) };

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

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::{OnnxError, OrtLoggingLevel};
    use crate::run_options::RunOptions;
    #[test]
    fn test_run_options() -> Result<(), OnnxError> {
        let mut opt = RunOptions::new()?;

        //verify default options
        assert!(!opt.should_terminate());
        assert_eq!(0, opt.log_verbosity_level());
        assert_eq!(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, opt.log_severity_level());
        assert_eq!("", opt.log_id()?);

        // try setting options
        opt.set_terminate(true)?;
        assert!(opt.should_terminate());

        opt.set_log_verbosity_level(1)?;
        assert_eq!(1, opt.log_verbosity_level());

        opt.set_log_severity_level(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)?;
        assert_eq!(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, opt.log_severity_level());

        opt.set_log_id("MyLogTag")?;
        assert_eq!("MyLogTag", opt.log_id()?);

        Ok(())
    }
}
