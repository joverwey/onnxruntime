extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-search=../../build/Windows/RelWithDebInfo/RelWithDebInfo");
        println!("cargo:rustc-link-search=../build/Windows/RelWithDebInfo/RelWithDebInfo");
    } else {
        println!("cargo:rustc-link-search=../../build/Linux/RelWithDebInfo");
    }

    println!("cargo:rustc-link-lib=dylib=onnxruntime");
    // println!("cargo:rustc-link-lib=static=onnxruntime_common");
    // println!("cargo:rustc-link-lib=static=onnxruntime_session");
    // println!("cargo:rustc-link-lib=static=onnxruntime_util");
    // println!("cargo:rustc-link-lib=static=onnxruntime_graph");
    // println!("cargo:rustc-link-lib=static=onnxruntime_mlas");
    // println!("cargo:rustc-link-lib=static=onnxruntime_optimizer");
    // println!("cargo:rustc-link-lib=static=onnxruntime_providers");
    // println!("cargo:rustc-link-lib=static=onnxruntime_framework");
    println!("cargo:rerun-if-changed=wrapper.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .whitelist_type("OrtApi")
        .whitelist_function("OrtGetApiBase")
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
