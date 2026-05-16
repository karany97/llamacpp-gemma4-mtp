# 02-swap-controller.tape
#
# Shows the swap-controller integration in action: register the MTP slot,
# swap from Q8 baseline to MTP, run a t/s probe on both, swap back.
#
# This is the "feature that only lives in this repo" demo — useful for
# the X thread that explains why the repo still exists post-merge.
#
# Run:
#   vhs demos/02-swap-controller.tape

Output demos/02-swap-controller.gif
Set FontSize 18
Set Width 1000
Set Height 600
Set Theme "Catppuccin Macchiato"
Set TypingSpeed 30ms
Set PlaybackSpeed 1.2

Type "cd /tmp/mtp-demo"
Enter
Sleep 500ms

# Inspect the integration drop-in
Type "head -40 scripts/swap_controller_integration.py"
Enter
Sleep 4s

# Show the slot registration
Type "python3 scripts/swap_controller_integration.py --register-only"
Enter
Sleep 3s

# Swap to MTP
Type "python3 scripts/swap_controller_integration.py --activate gemma-mtp"
Enter
Sleep 3s

# Probe t/s
Type "curl -s http://localhost:8005/v1/chat/completions \\"
Enter
Type "  -H 'Content-Type: application/json' \\"
Enter
Type "  -d '{\"model\":\"gemma-mtp\",\"messages\":[{\"role\":\"user\",\"content\":\"summarise the speed of sound\"}],\"max_tokens\":100}' | jq .usage"
Enter
Sleep 5s

# Swap back to baseline
Type "python3 scripts/swap_controller_integration.py --activate gemma-baseline"
Enter
Sleep 2s

Type "echo 'swap controller drill done. exit:' $?"
Enter
Sleep 2s
