use std::{env, ffi::OsString, path::PathBuf, process::Command};

fn main() {
    link_apple_clang_runtime();
    tauri_build::build()
}

fn link_apple_clang_runtime() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    let resource_dir = apple_clang_path()
        .and_then(clang_resource_dir)
        .or_else(|| clang_resource_dir(env::var_os("CC").unwrap_or_else(|| OsString::from("cc"))));
    let Some(resource_dir) = resource_dir else {
        println!("cargo:warning=Apple Clang resource directory was unavailable; skipping compiler runtime linkage");
        return;
    };

    let runtime_dir = resource_dir.join("lib").join("darwin");
    if runtime_dir.join("libclang_rt.osx.a").is_file() {
        println!("cargo:rustc-link-search=native={}", runtime_dir.display());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    } else {
        println!("cargo:warning=Apple Clang runtime archive was unavailable; skipping compiler runtime linkage");
    }
}

fn apple_clang_path() -> Option<OsString> {
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", "clang"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!path.is_empty()).then_some(path.into())
}

fn clang_resource_dir(compiler: OsString) -> Option<PathBuf> {
    let output = Command::new(compiler)
        .arg("-print-resource-dir")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!path.is_empty()).then_some(PathBuf::from(path))
}
