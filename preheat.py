import sys
import os
import time

# Add project root to path to ensure indextts can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))

print("[Preheat] Starting background preheat of Torch/Transformers...")
sys.stdout.flush()

try:
    # 1. Import Torch
    import torch
    print("[Preheat] Torch imported.")
    sys.stdout.flush()

    # 2. Import Transformers
    import transformers
    print("[Preheat] Transformers imported.")
    sys.stdout.flush()

    # 3. Import VoxAI (IndexTTS2) Engine
    # This might fail if dependencies aren't perfect, but we try anyway
    try:
        from indextts.infer_v2 import IndexTTS2
        print("[Preheat] VoxAI (IndexTTS2) engine imported.")
    except Exception as e:
        print(f"[Preheat] VoxAI import skipped: {e}")
    sys.stdout.flush()
    
except Exception as e:
    print(f"[Preheat] Error during preheat: {e}")

print("[Preheat] Done! File system cache warmed up.")
sys.exit(0)
