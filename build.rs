use librarian;

fn main() {
    if cfg!(windows) {
        Command::new("build.bat").status().unwrap();
    } else {
        Command::new("build.sh").status().unwrap();
    }
}
