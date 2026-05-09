"""Drop-in integration for a `llm_swap_controller.py`-style swap controller
to add a 'gemma-mtp' slot that runs ik_llama.cpp with PR #1744 + Gemma 4 MTP.

Two deployment patterns are covered:

  (A) NATIVE LINUX / WSL2 host — the swap controller runs on the same Linux
      kernel that hosts the patched llama-server. Args dispatch by adding
      five flags to your existing arg list.

  (B) WINDOWS HOST + WSL2 BACKEND — the swap controller is on Windows
      (e.g. running natively to drive a Windows-native baseline llama-server),
      but our patched binary lives inside WSL2 because Windows lacks the
      Visual Studio Build Tools and CUDA Toolkit needed for a native build.
      We launch via `wsl.exe --user root -d <distro> -- bash -lc '<cmd>'`,
      and rely on WSL2's automatic localhost forwarding (Win11 22H2+) so the
      swap controller's health-check against http://127.0.0.1:8005/health
      keeps working unchanged.

The functions below return a list of args ready to pass to subprocess /
WMI Win32_Process.Create / etc.
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Pattern A: native Linux / WSL2 host
# ────────────────────────────────────────────────────────────────────────────

def build_mtp_launch_args_linux(cfg: dict, port: int) -> list[str]:
    """Generate launch args for ik_llama.cpp + Gemma 4 MTP drafter.

    Mirrors the /scripts/run_bench.sh recipe — official PR 1744 flags.

    Required cfg keys:
      llama_bin     absolute path to the patched llama-server binary
      model_path    absolute path to the target GGUF (Gemma 4 31B Q8_0)
      drafter_path  absolute path to the drafter GGUF (Gemma 4 31B assistant Q8_0)
      alias         OpenAI-compat model name reported in /v1/models
      context       int — context length (16384/32768/65536 typical)
      threads       int — CPU thread count for prompt prefill / sampling
      parallel      int — `-np` (1 for interactive)
      draft_max     int — speculative lookahead, 3 is the sweet spot
      draft_p_min   float — drafter min-prob filter (0.0 = lossless)
    """
    return [
        cfg["llama_bin"],
        "-m", cfg["model_path"],
        "--port", str(port),
        "--host", "0.0.0.0",
        "-ngl", "999",
        "-c", str(cfg["context"]),
        "-fa", "on",
        "-np", str(cfg["parallel"]),
        "--threads", str(cfg["threads"]),
        "--mlock",
        "--jinja",
        "--tensor-split", "1,1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        # === MTP-specific ===
        "--spec-type", "mtp",
        "-md", cfg["drafter_path"],
        "-ngld", "99",
        "--draft-max", str(cfg["draft_max"]),
        "--draft-p-min", str(cfg["draft_p_min"]),
        # ====================
        "--alias", cfg["alias"],
    ]


# ────────────────────────────────────────────────────────────────────────────
# Pattern B: Windows host, binary inside WSL2
# ────────────────────────────────────────────────────────────────────────────

def windows_to_wsl_path(win_path: str) -> str:
    r"""Translate a Windows path like 'A:\models\foo.gguf' to '/mnt/a/models/foo.gguf'.

    WSL2 auto-mounts Windows drives under /mnt/<lowercase-letter>/ by default
    (see /etc/wsl.conf [automount]). This helper covers the common shape and
    backslash-to-forward-slash conversion. UNC paths and drive-less paths are
    returned unchanged — caller's responsibility.
    """
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path.replace("\\", "/")


def build_mtp_wsl_command(cfg: dict, port: int,
                           wsl_distro: str = "Ubuntu-24.04",
                           wsl_user: str = "root") -> str:
    r"""Generate the SINGLE Windows-cmd-line string that launches the patched
    llama-server inside WSL2.

    Use this string with WMI Win32_Process.Create or with the existing
    `_launch_via_wmi` helper in your `llm_swap_controller.py`. It survives
    the parent process exiting because wsl.exe forks and `bash -lc` keeps
    running until killed.

    Note: paths in cfg are Windows paths (A:\models\...). We translate
    them to WSL /mnt/a/... paths for use inside the bash command.
    """
    bin_inside_wsl = cfg["llama_bin_wsl"]      # e.g. /root/llamacpp-gemma4-mtp/build/.../bin/llama-server
    target_inside  = windows_to_wsl_path(cfg["model_path"])
    drafter_inside = windows_to_wsl_path(cfg["drafter_path"])

    bash_args = (
        f"{bin_inside_wsl} "
        f"-m '{target_inside}' "
        f"--port {port} --host 0.0.0.0 "
        f"-ngl 999 -c {cfg['context']} -fa on "
        f"-np {cfg['parallel']} --threads {cfg['threads']} "
        f"--mlock --jinja --tensor-split 1,1 "
        f"--cache-type-k q8_0 --cache-type-v q8_0 "
        f"--spec-type mtp -md '{drafter_inside}' -ngld 99 "
        f"--draft-max {cfg['draft_max']} --draft-p-min {cfg['draft_p_min']} "
        f"--alias {cfg['alias']}"
    )
    # Windows command line: wsl.exe forks once and the WSL session keeps the
    # llama-server child alive even after wsl.exe returns to caller.
    return (
        f'wsl.exe --user {wsl_user} -d {wsl_distro} -- '
        f'bash -lc "{bash_args}"'
    )


# ────────────────────────────────────────────────────────────────────────────
# Reference: NandaiJarvis-style MODELS dict entry
# ────────────────────────────────────────────────────────────────────────────

NANDAI_GEMMA_MTP_SLOT = {
    "alias":       "gemma-4-31b-mtp",
    "label":       "Gemma 4 31B Q8 + MTP drafter (PR 1744, ~2.6x lossless)",
    "model_path":  r"A:\models\gemma-4-31B-it-Q8_0.gguf",
    "drafter_path": r"A:\models\gemma-4-31B-it-assistant-Q8_0.gguf",
    "context":     65536,
    "threads":     12,
    "parallel":    1,
    "engine":      "ik_llama_cpp_mtp_wsl",
    "llama_bin_wsl": "/root/llamacpp-gemma4-mtp/build/ik_llama.cpp/build/bin/llama-server",
    "draft_max":   3,
    "draft_p_min": 0.0,
}


# ────────────────────────────────────────────────────────────────────────────
# Drop-in patch for `_launch_model()` dispatch
# ────────────────────────────────────────────────────────────────────────────
#
# In your existing `_launch_model(name)`, replace the args build with this:
#
#   cfg = MODELS[name]
#   if not Path(cfg["model_path"]).exists():
#       raise FileNotFoundError(f"GGUF not found: {cfg['model_path']}")
#
#   engine = cfg.get("engine", "llama_cpp")
#   if engine == "ik_llama_cpp_mtp_wsl":
#       cmd_line = build_mtp_wsl_command(cfg, port=INFERENCE_PORT)
#   elif engine == "ik_llama_cpp_mtp":          # Linux/WSL native swap controller
#       args = build_mtp_launch_args_linux(cfg, port=INFERENCE_PORT)
#       cmd_line = " ".join(f'"{a}"' if " " in a else a for a in args)
#   else:                                        # existing llama.cpp baseline
#       args = [
#           f'"{LLAMA_BIN}"', f'-m "{cfg["model_path"]}"',
#           # ... (your existing baseline args) ...
#       ]
#       cmd_line = " ".join(args)
#
#   bat_content = f'@echo off\r\n{cmd_line} > "{LAUNCH_OUT}" 2> "{LAUNCH_ERR}"\r\n'
#   LAUNCH_BAT.write_text(bat_content, encoding="ascii")
#   _kill_llama_server()
#   pid = _launch_via_wmi(f"cmd.exe /c {LAUNCH_BAT}")
#   PID_FILE.write_text(str(pid), encoding="ascii")
#   _write_active(name)
#   return pid
#
# Then add NANDAI_GEMMA_MTP_SLOT to your MODELS dict (rename the key to
# "gemma-mtp" or whatever fits your naming scheme).


if __name__ == "__main__":
    import json

    print("=== Pattern A: native Linux/WSL2 host ===")
    cfg_linux = {
        "llama_bin":   "/root/llamacpp-gemma4-mtp/build/ik_llama.cpp/build/bin/llama-server",
        "model_path":  "/mnt/a/models/gemma-4-31B-it-Q8_0.gguf",
        "drafter_path": "/mnt/a/models/gemma-4-31B-it-assistant-Q8_0.gguf",
        "alias":       "gemma-4-31b-mtp",
        "context":     65536,
        "threads":     12,
        "parallel":    1,
        "draft_max":   3,
        "draft_p_min": 0.0,
    }
    args = build_mtp_launch_args_linux(cfg_linux, port=8005)
    print(" ".join(args))

    print()
    print("=== Pattern B: Windows swap controller, binary in WSL ===")
    cmd = build_mtp_wsl_command(NANDAI_GEMMA_MTP_SLOT, port=8005)
    print(cmd)

    print()
    print("=== NANDAI_GEMMA_MTP_SLOT (MODELS dict entry) ===")
    print(json.dumps(NANDAI_GEMMA_MTP_SLOT, indent=2))
