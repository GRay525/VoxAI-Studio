import time
import sys

def profile_import(module_name):
    print(f"DEBUG: Profiling import of {module_name}...")
    start = time.perf_counter()
    try:
        __import__(module_name)
        duration = time.perf_counter() - start
        print(f"SUCCESS: {module_name} imported in {duration:.2f}s")
    except Exception as e:
        print(f"ERROR: {module_name} failed to import: {e}")

if __name__ == "__main__":
    print(f"Python Version: {sys.version}")
    profile_import("torch")
    profile_import("transformers")
    profile_import("librosa")
    
    print("DEBUG: Profiling indextts.infer_v2...")
    start = time.perf_counter()
    try:
        from indextts.infer_v2 import IndexTTS2
        print(f"SUCCESS: IndexTTS2 imported in {time.perf_counter() - start:.2f}s")
    except Exception as e:
        print(f"ERROR: IndexTTS2 failed to import: {e}")
