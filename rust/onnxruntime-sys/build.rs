extern crate bindgen;

use librarian::{add_link_search_path, download_or_find_file, extract_archive, ExtractError};
use std::env;
use std::path::PathBuf;

fn download_dependencies() -> Result<PathBuf, ExtractError> {
    let (path, folder) = if cfg!(target_os = "windows") {
        ("https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-win-x64-1.4.0.zip", "onnxruntime-win-x64-1.4.0")
    } else if cfg!(target_os = "linux") {
        ("https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz", "onnxruntime-linux-x64-1.4.0")
    } else if cfg!(target_os = "macos") {
        ("https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-osx-x64-1.4.0.tgz", "onnxruntime-osx-x64-1.4.0")
    } else {
        panic!("Unsupported platform")
    };

    let path_to_lib_zip = download_or_find_file(path.into(), None).expect(&format!(
        "Unable to download pre-compiled binaries from '{}'",
        path
    ));

    let path_to_extracted_files = extract_archive(&path_to_lib_zip, None)?.join(folder);

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
    librarian::install_dylibs(&path_to_dylib_folder, None, Some(&deps_folder))
        .expect("Failed to install dynamic libraries");
    librarian::install_dylibs(&path_to_dylib_folder, None, Some(&examples_folder))
        .expect("Failed to install dynamic libraries");

    add_link_search_path(&path_to_dylib_folder);
    // if cfg!(windows) {
    //     println!("cargo:rustc-link-search=../../build/Windows/RelWithDebInfo/RelWithDebInfo");
    //     println!("cargo:rustc-link-search=build/Windows/RelWithDebInfo/RelWithDebInfo");
    // } else {
    //     println!("cargo:rustc-link-search=../../build/Linux/RelWithDebInfo");
    //     println!("cargo:rustc-link-search=build/Linux/RelWithDebInfo");
    // }

    println!("cargo:rustc-link-lib=dylib=onnxruntime");
    let bindings = bindgen::Builder::default()
        .header(
            path_to_include_folder
                .join("onnxruntime_c_api.h")
                .to_string_lossy(),
        )
        .whitelist_type("OrtApi")
        .whitelist_function("OrtGetApiBase")
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
