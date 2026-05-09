"""Drop-in patch for a NandaiJarvis-style llm_swap_controller.py to add a
'gemma-mtp' slot that runs ik_llama.cpp with PR #1744 + Gemma 4 MTP drafter.

Usage:
    1. Build ik_llama.cpp via scripts/apply_patches.sh + scripts/build_cuda_*.sh.
       The new llama-server.exe lives at build/ik_llama.cpp/build/bin/llama-server[.exe].
    2. Drop the dict below into your MODELS registry in llm_swap_controller.py:

       MODELS["gemma-mtp"] = {
           "alias":       "gemma-4-31b-mtp",
           "label":       "Gemma 4 31B Q8 + MTP drafter (PR 1744, ~2.6x lossless)",
           "model_path":  r"A:\\models\\gemma-4-31B-it-Q8_0.gguf",
           "drafter_path":r"A:\\models\\gemma-4-31B-it-assistant-Q8_0.gguf",
           "context":     65536,
           "threads":     12,
           "parallel":    1,
           "engine":      "ik_llama_cpp_mtp",
           "llama_bin":   r"A:\\ik_llama.cpp\\build\\bin\\llama-server.exe",
           "draft_max":   3,
           "draft_p_min": 0.0,
       }

    3. In _launch_model(), branch on cfg.get("engine") and emit the MTP-aware
       command line below when engine == "ik_llama_cpp_mtp".
"""

def build_mtp_launch_args(cfg: dict, port: int) -> list[str]:
    """Generate launch args for ik_llama.cpp + Gemma 4 MTP drafter.

    Mirrors the /scripts/run_bench.sh recipe — official PR 1744 flags.
    Returns a list of args ready to pass to subprocess / WMI Win32_Process.Create.
    """
    return [
        f'"{cfg["llama_bin"]}"',
        f'-m "{cfg["model_path"]}"',
        f'--port {port}',
        f'--host 0.0.0.0',
        f'-ngl 999',
        f'-c {cfg["context"]}',
        f'-fa on',
        f'-np {cfg["parallel"]}',
        f'--threads {cfg["threads"]}',
        f'--mlock',
        f'--jinja',
        f'--tensor-split 1,1',
        f'--cache-type-k q8_0',
        f'--cache-type-v q8_0',
        # === MTP-specific ===
        f'--spec-type mtp',
        f'-md "{cfg["drafter_path"]}"',
        f'-ngld 99',
        f'--draft-max {cfg["draft_max"]}',
        f'--draft-p-min {cfg["draft_p_min"]}',
        # ====================
        f'--alias {cfg["alias"]}',
    ]


# Reference for testing: the analogous llama.cpp non-MTP launch args (current
# baseline) live in NandaiJarvis llm_swap_controller.py:
#
#   args = [
#       f'"{LLAMA_BIN}"', f'-m "{cfg["model_path"]}"', f'--port {INFERENCE_PORT}',
#       f'--host 0.0.0.0', f'-ngl 999', f'-c {cfg["context"]}', f'-fa on',
#       f'-np {cfg["parallel"]}', f'--threads {cfg["threads"]}', f'--mlock',
#       f'--jinja', f'--tensor-split 1,1', f'--cache-type-k q8_0',
#       f'--cache-type-v q8_0', f'--alias {cfg["alias"]}',
#   ]
#
# The MTP version adds 5 lines: --spec-type mtp / -md / -ngld 99 / --draft-max /
# --draft-p-min. Everything else is identical, so the kill/launch/PID-tracking
# logic in the swap controller doesn't change.


if __name__ == "__main__":
    # Demo the args this would emit
    demo_cfg = {
        "alias":       "gemma-4-31b-mtp",
        "label":       "Gemma 4 31B Q8 + MTP",
        "model_path":  r"A:\models\gemma-4-31B-it-Q8_0.gguf",
        "drafter_path":r"A:\models\gemma-4-31B-it-assistant-Q8_0.gguf",
        "context":     65536,
        "threads":     12,
        "parallel":    1,
        "engine":      "ik_llama_cpp_mtp",
        "llama_bin":   r"A:\ik_llama.cpp\build\bin\llama-server.exe",
        "draft_max":   3,
        "draft_p_min": 0.0,
    }
    print(" ".join(build_mtp_launch_args(demo_cfg, port=8005)))
