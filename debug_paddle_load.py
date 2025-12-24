"""
debug_paddle_load.py

Diagnose which native DLL or extension fails when importing `paddle` on Windows.

Usage (in the same conda env that fails):
  conda activate ptToonnx
  python debug_paddle_load.py

This script:
- Adds conda nvidia DLL folders (cudnn, cuda_runtime, cublas) via os.add_dll_directory
- Attempts to load all *.dll under those folders using LoadLibraryExW then LoadLibraryW,
  reporting success/failure and Windows error codes.
- Attempts to load all .pyd/.dll files under site-packages/paddle (these are paddle native extensions).
- Attempts to import paddle and prints traceback and ctypes.get_last_error().
- Writes a detailed log file debug_paddle_load.log in current directory.

Note: run this with the same python executable that later fails to import paddle.
"""
from __future__ import annotations
import os
import sys
import glob
import ctypes
import traceback
import datetime

LOGFILE = "debug_paddle_load.log"

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log(s: str):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def log(s: str):
    line = f"[{now()}] {s}"
    print(line)
    write_log(line)

def safe_add_dll_directory(p: str):
    try:
        if os.path.isdir(p):
            os.add_dll_directory(p)
            log(f"Added DLL directory: {p}")
        else:
            log(f"Directory not found (skip): {p}")
    except Exception as e:
        log(f"add_dll_directory FAILED for {p}: {e}")
        traceback.print_exc()

def try_load_dlls(dll_list):
    kernel32 = ctypes.windll.kernel32
    results = []
    prev_path = os.environ.get("PATH", "")
    for dll in dll_list:
        basename = os.path.basename(dll)
        log("-" * 80)
        log(f"Attempt loading: {dll}")
        try:
            # Try LoadLibraryExW first (like paddle does) with flags 0x00001100
            res = kernel32.LoadLibraryExW(ctypes.c_wchar_p(dll), None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res:
                log(f"LoadLibraryExW SUCCESS: {dll} (handle: {res})")
                try:
                    kernel32.FreeLibrary(res)
                except Exception:
                    pass
                results.append((dll, "LoadLibraryExW_OK", 0))
                continue
            else:
                log(f"LoadLibraryExW returned NULL, last_error={last_error}")
            # Try patching PATH temporarily (prepend dll folder) and LoadLibraryW
            folder = os.path.dirname(dll)
            os.environ["PATH"] = os.pathsep.join([folder, prev_path])
            res2 = kernel32.LoadLibraryW(ctypes.c_wchar_p(dll))
            last_error2 = ctypes.get_last_error()
            if res2:
                log(f"LoadLibraryW SUCCESS: {dll} (handle: {res2})")
                try:
                    kernel32.FreeLibrary(res2)
                except Exception:
                    pass
                results.append((dll, "LoadLibraryW_OK", 0))
            else:
                log(f"LoadLibraryW returned NULL, last_error={last_error2}")
                # Turn code into readable exception
                try:
                    raise ctypes.WinError(last_error2)
                except Exception as e:
                    log(f"CTypes WinError for {dll}: {e}")
                    results.append((dll, "FAIL", int(last_error2)))
                    # include full traceback text to log
                    tb = traceback.format_exc()
                    write_log(tb)
        except Exception as e:
            log(f"Exception while loading {dll}: {repr(e)}")
            tb = traceback.format_exc()
            write_log(tb)
            results.append((dll, "EXC", None))
        finally:
            # restore PATH for next iteration
            os.environ["PATH"] = prev_path
    return results

def find_dlls_under_dirs(dirs):
    dlls = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        globbed = glob.glob(os.path.join(d, "*.dll"))
        dlls.extend(globbed)
    # filter out obvious 32-bit naming used by some packages
    dlls = [p for p in dlls if "32_" not in os.path.basename(p).lower()]
    # unique ordered
    seen = set()
    out = []
    for p in dlls:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def find_paddle_extensions(paddle_pkg_dir):
    patterns = ["*.dll", "*.pyd"]
    found = []
    if not os.path.isdir(paddle_pkg_dir):
        return found
    for root, _, files in os.walk(paddle_pkg_dir):
        for f in files:
            if f.lower().endswith(".dll") or f.lower().endswith(".pyd"):
                found.append(os.path.join(root, f))
    return found

def main():
    log("=== debug_paddle_load START ===")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version}")
    log(f"Current working dir: {os.getcwd()}")
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix or ""
    log(f"CONDA_PREFIX (detected): {conda_prefix}")

    # build candidate nvidia/cuda dirs from conda env
    cand_dirs = []
    if conda_prefix:
        cand_dirs.extend([
            os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
            os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
            os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"),
            os.path.join(conda_prefix, "Lib", "site-packages", "nvidia", "nvjitlink", "bin"),
        ])
    # also try system CUDA (common install)
    possible_sys_cuda = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    ]
    cand_dirs.extend([d for d in possible_sys_cuda if os.path.isdir(d)])

    # add and log
    for p in cand_dirs:
        safe_add_dll_directory(p)

    # collect DLLs
    dlls = find_dlls_under_dirs(cand_dirs)
    log(f"Found {len(dlls)} DLLs to test under candidate dirs.")
    for d in dlls:
        log(f" - {d}")

    # Try loading these DLLs
    dll_results = try_load_dlls(dlls)

    # Inspect paddle extension .pyd/.dll
    site_packages = os.path.join(conda_prefix, "Lib", "site-packages") if conda_prefix else None
    paddle_pkg = None
    if site_packages:
        paddle_pkg = os.path.join(site_packages, "paddle")
    if paddle_pkg and os.path.isdir(paddle_pkg):
        log(f"Scanning paddle package directory for extensions: {paddle_pkg}")
        paddle_exts = find_paddle_extensions(paddle_pkg)
        log(f"Found {len(paddle_exts)} paddle extension files (.pyd/.dll).")
        for p in paddle_exts:
            log("Paddle ext candidate: " + p)
    else:
        log("Paddle package directory not found at expected location: " + str(paddle_pkg))
        paddle_exts = []

    # Try to load paddle extension binaries directly
    if paddle_exts:
        log("Attempting to load paddle extension binaries directly...")
        ext_results = try_load_dlls(paddle_exts)
    else:
        ext_results = []

    # Now attempt to import paddle and capture last error and traceback
    log("Now trying: import paddle")
    try:
        import importlib
        # ensure any previous state is cleared (careful)
        if "paddle" in sys.modules:
            del sys.modules["paddle"]
        import paddle  # actual import
        log("IMPORT paddle: SUCCESS")
        importlib.reload(paddle)
        log("paddle module reloaded OK")
        import_ok = True
        import_err = None
        last_error = ctypes.get_last_error()
    except Exception as e:
        import_ok = False
        import_err = traceback.format_exc()
        last_error = ctypes.get_last_error()
        log("IMPORT paddle: FAILED")
        log(f"Exception: {e}")
        write_log(import_err)
        log(f"ctypes.get_last_error() = {last_error}")

    # Summarize
    log("=== SUMMARY ===")
    log(f"Tested DLLs count: {len(dlls)}")
    failed = [r for r in dll_results if r[1] != "LoadLibraryExW_OK" and r[1] != "LoadLibraryW_OK"]
    log(f"Failed native nvidia DLLs: {len(failed)} (see log for details)")
    for f in failed[:50]:
        log(f" FAIL: {f}")
    log(f"Paddle extension binaries tested: {len(paddle_exts)}")
    ext_failed = [r for r in ext_results if r[1] != "LoadLibraryExW_OK" and r[1] != "LoadLibraryW_OK"]
    log(f"Failed paddle extension loads: {len(ext_failed)}")
    for f in ext_failed[:50]:
        log(f" PEXT FAIL: {f}")
    log(f"Import paddle success?: {import_ok}")
    if not import_ok:
        log(f"paddle import traceback saved to {LOGFILE}")
        log(f"ctypes.get_last_error() after import attempt: {last_error}")

    log("=== debug_paddle_load END ===")
    log(f"Detailed log written to {os.path.abspath(LOGFILE)}")

if __name__ == "__main__":
    # on windows only
    if os.name != "nt":
        print("This script is intended to run on Windows.")
        sys.exit(1)
    # clear log file
    try:
        open(LOGFILE, "w", encoding="utf-8").close()
    except Exception:
        pass
    main()