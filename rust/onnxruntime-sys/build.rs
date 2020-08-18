extern crate bindgen;

use librarian::{add_link_search_path, download_or_find_file, extract_archive, ExtractError};
use std::path::PathBuf;
use std::{env, fs};

fn download_dependencies() -> Result<PathBuf, ExtractError> {
    let version = env::var("ONNX_VERSION").unwrap_or("1.4.0".to_string());

    let (platform, extension) = if cfg!(target_os = "windows") {
        ("win", "zip")
    } else if cfg!(target_os = "linux") {
        ("linux", "tgz")
    } else if cfg!(target_os = "macos") {
        ("osx", "tgz")
    } else {
        panic!("Unsupported platform")
    };

    let gpu = if cfg!(feature = "gpu") && !cfg!(target_os = "macos") {
        "-gpu"
    } else {
        ""
    };

    let folder = format!("onnxruntime-{}-x64{}-{}", platform, gpu, version);

    let test_model_file = "squeezenet.onnx";
    let test_model_url = format!(
        "https://github.com/microsoft/onnxruntime/raw/master/csharp/testdata/{}",
        test_model_file
    );

    let archive_file = format!(
        "onnxruntime-{}-x64{}-{}.{}",
        platform, gpu, version, extension
    );
    let archive_url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}/{}",
        version, archive_file
    );

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archive_target_file = out_path.join(archive_file);

    let mut test_folder = out_path.clone();

    while test_folder.pop() && !test_folder.ends_with("target") {}
    test_folder.push("data");

    let model_target_file = test_folder.join(test_model_file);

    if !model_target_file.exists() {
        fs::create_dir_all(&test_folder).expect("Unable to create test directory");
        if let Err(_) = download_or_find_file(&test_model_url, Some(&test_folder)) {
            panic!(
                "Unable to download test file '{}'. Unit tests depending on this model will fail.",
                test_model_url
            );
        }
    }

    let path_to_lib_zip = if !archive_target_file.exists() {
        download_or_find_file(&archive_url, None).expect(&format!(
            "Unable to download pre-compiled binaries from '{}'",
            archive_url
        ))
    } else {
        archive_target_file
    };

    let path_to_extracted_files = out_path.join(&folder);
    if !path_to_extracted_files.exists() {
        extract_archive(&path_to_lib_zip, None)?;
    }

    Ok(path_to_extracted_files)
}

fn main() {
    let path_to_extracted_files =
        download_dependencies().expect("Unable to download and extract dependencies");
    let path_to_dylib_folder = path_to_extracted_files.join("lib");
    let path_to_include_folder = path_to_extracted_files.join("include");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut target_folder = out_path.clone();

    while !target_folder.ends_with("debug") && !target_folder.ends_with("release") {
        target_folder.pop();
    }

    let deps_folder = target_folder.join("deps");
    let examples_folder = target_folder.join("examples");
    println!(
        "path_to_dylib_folder:{}",
        path_to_dylib_folder.to_string_lossy()
    );
    librarian::install_dylibs(&path_to_dylib_folder, None, Some(&deps_folder))
        .expect("Failed to install dynamic libraries");
    librarian::install_dylibs(&path_to_dylib_folder, None, Some(&examples_folder))
        .expect("Failed to install dynamic libraries");

    add_link_search_path(&path_to_dylib_folder);

    println!("cargo:rustc-link-lib=dylib=onnxruntime");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", path_to_include_folder.to_string_lossy()))
        .whitelist_type("OrtApi")
        .whitelist_function("OrtGetApiBase")
        .whitelist_function("OrtSessionOptionsAppendExecutionProvider_CUDA")
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
