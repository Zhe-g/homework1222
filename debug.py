# test_load_all.py
import os, glob, ctypes, traceback

conda = os.environ.get("CONDA_PREFIX") or r"C:\Users\GuoZhe\.conda\envs\ptToonnx"
dll_dirs = [
    os.path.join(conda, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(conda, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    os.path.join(conda, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(conda, "Lib", "site-packages", "nvidia", "cublas", "bin"),  # 如需扩展，可加其它目录
]

# 收集所有 dll（按 paddle 那段代码逻辑）
dlls = []
for d in dll_dirs:
    if os.path.isdir(d):
        dlls.extend(glob.glob(os.path.join(d, "*.dll")))

# 过滤 32-bit 名称（同 paddle 的做法）
dlls = [p for p in dlls if "32_" not in os.path.basename(p).lower()]

kernel32 = ctypes.windll.kernel32
prev_error_mode = kernel32.SetErrorMode(0)  # 不改变太多，简单试
print("总共检测 DLL 数量:", len(dlls))

for dll in dlls:
    print("="*80)
    print("尝试加载:", dll)
    try:
        # 先尝试 LoadLibraryExW (flags 如 paddle 使用 0x00001100)
        res = kernel32.LoadLibraryExW(ctypes.c_wchar_p(dll), None, 0x00001100)
        last_error = ctypes.get_last_error()
        if res:
            print("LoadLibraryExW 成功:", dll)
            # 若加载成功，free 掉句柄以免占用
            kernel32.FreeLibrary(res)
            continue
        else:
            print("LoadLibraryExW 返回 NULL, last_error =", last_error)
        # 再尝试临时修改 PATH 的方法（把目录前置）
        prev_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(dll), prev_path])
        try:
            res2 = kernel32.LoadLibraryW(ctypes.c_wchar_p(dll))
            last_error2 = ctypes.get_last_error()
            if res2:
                print("LoadLibraryW 成功:", dll)
                kernel32.FreeLibrary(res2)
            else:
                print("LoadLibraryW 返回 NULL, last_error =", last_error2)
                # 打印可读错误
                try:
                    raise ctypes.WinError(last_error2)
                except Exception as e:
                    traceback.print_exc()
        finally:
            os.environ["PATH"] = prev_path
    except Exception as e:
        print("异常:", repr(e))
        traceback.print_exc()