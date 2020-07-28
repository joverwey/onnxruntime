fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-search=../../build/Windows/RelWithDebInfo/RelWithDebInfo");
        println!("cargo:rustc-link-search=build/Windows/RelWithDebInfo/RelWithDebInfo");
    } else {
        println!("cargo:rustc-link-search=../../build/Linux/RelWithDebInfo");
        println!("cargo:rustc-link-search=build/Linux/RelWithDebInfo");
    }
}
