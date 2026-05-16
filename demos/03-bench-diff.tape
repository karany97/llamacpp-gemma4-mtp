# 03-bench-diff.tape
#
# Shows the value-prop: `diff` your fresh bench against the PR-1744
# author's reference JSON. If the speedup matches within ±10% you've
# successfully reproduced the published numbers on your box.
#
# This is the tweet-sized demo for the X thread.

Output demos/03-bench-diff.gif
Set FontSize 18
Set Width 1000
Set Height 500
Set Theme "Catppuccin Macchiato"
Set TypingSpeed 30ms
Set PlaybackSpeed 1.4

Type "cd /tmp/mtp-demo"
Enter
Sleep 500ms

# Show reference
Type "jq '{hardware:.hardware.cpu, speedup, acceptance:.with_mtp.acceptance_rate}' benchmarks/pr-1744-author.json"
Enter
Sleep 3s

# Show fresh local bench
Type "jq '{hardware:.hardware.cpu, speedup, acceptance:.with_mtp.acceptance_rate}' benchmarks/$(hostname)-*.json"
Enter
Sleep 3s

# Diff just the headline number
Type "echo -n 'Reference speedup: ' && jq .speedup benchmarks/pr-1744-author.json"
Enter
Type "echo -n 'My box speedup:    ' && jq .speedup benchmarks/$(hostname)-*.json"
Enter
Sleep 4s

Type "echo 'reproduction verified — pin this JSON in your CI to catch drift'"
Enter
Sleep 3s
