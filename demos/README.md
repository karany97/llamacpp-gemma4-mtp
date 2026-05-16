# Demo recordings

Two-tool pipeline for capturing repeatable terminal demos. Both tools
produce embeddable output (GIF / cast file) that drops into the README,
release notes, or a social post without an editing pass.

## Why VHS for the terminal demos

VHS by Charmbracelet ([github.com/charmbracelet/vhs](https://github.com/charmbracelet/vhs))
records terminal sessions from a declarative `.tape` script. Same script,
same output, every time — which matters because the build journal's
"no cheating" rule says every published number traces to a reproducible
run. A `.tape` IS the reproducible run.

Install on macOS:
```bash
brew install vhs
```

Install on Linux:
```bash
go install github.com/charmbracelet/vhs@latest
# OR via the Charm release:
curl -sL https://github.com/charmbracelet/vhs/releases/latest/download/vhs_Linux_x86_64.tar.gz | tar -xz
```

## Why asciinema as the fallback

asciinema ([asciinema.org](https://asciinema.org)) is the no-install,
text-only option. The cast file is plain JSON, embeds in any markdown
via the player widget, and survives forever on asciinema.org (or as a
file in this repo).

Install:
```bash
brew install asciinema  # OR: pip install asciinema
```

## Capturing a demo

### With VHS (preferred for new captures)

```bash
vhs demos/01-quick-start.tape          # outputs demos/01-quick-start.gif
```

The output GIF embeds directly in the README:

```markdown
![Quick start](./demos/01-quick-start.gif)
```

### With asciinema (preferred for embed-in-blog)

```bash
asciinema rec demos/01-quick-start.cast --command './scripts/run_bench.sh'
# Upload to asciinema.org if you want a hosted player:
asciinema upload demos/01-quick-start.cast
```

## Inventory

| File | Captures | Best for |
|---|---|---|
| `01-quick-start.tape` | `apply_patches.sh && build_cpu.sh && run_bench.sh` end-to-end | README hero GIF |
| `02-swap-controller.tape` | `swap_controller_integration.py` in action | Release notes |
| `03-bench-diff.tape` | `diff` between author's reference JSON and a fresh run | Tweet / HN comment |

## Style guide

- Width 1000 px, height 600 px (good for GitHub README + tweet embed)
- Theme: `Catppuccin Macchiato` (warm, easy to read in light + dark mode)
- Font: `JetBrains Mono`, 18 px
- TypingSpeed: `30ms` (slow enough to follow, fast enough not to bore)
- Total runtime: < 90 seconds — anything longer loses the social-post audience
- Always show the exit code on completion (`echo $?`) so viewers see the run succeeded

## Where the captured GIFs / casts live

In the `demos/` folder, committed. Not in releases (those are for binary
artifacts only). README references them by relative path so they work
offline + on forks.
