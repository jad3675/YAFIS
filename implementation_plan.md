# Implementation Plan (Current State)

This file tracks what has already been implemented vs what remains, based on the work done in this repo so far.

## What’s been implemented (current)

### 1) Logging / error handling cleanup (partial)
- Centralized logging is used across most modules via [`negative_converter.utils.logger.get_logger()`](negative_converter/utils/logger.py:1).
- Several noisy `print(...)` error paths were removed and replaced with structured logging (where applicable).

### 2) Preset correctness fixes (applied)
- Fixed a “film” photo-style preset rendering black due to grain scaling mismatch (0..255 vs 0..1 expected).
- Implemented missing application of `colorGrading` in the photo preset pipeline.

### 3) UI & UX improvements (applied)
- Preset panels were tightened:
  - Film Simulation and Photo Styles panels were wrapped in scroll areas to avoid large unused whitespace when preset lists are long.
  - Redundant “Preview” buttons were removed where single-click selection already previews.
- Added **wipe compare slider** in the View toolbar (“Compare: Before … After”), and kept the original compare toggle.
- Added throttling/debouncing to the compare slider to reduce repaint storms.

### 4) Performance improvements (partial)
- Histogram updates are debounced and computed using a downscaled proxy image to reduce UI stalls (in the histogram widget).
- A known UI-thread bottleneck was removed:
  - The compare slider previously recomputed a full “after” image on the UI thread when set to 100%; that recompute was removed.

### 5) Color management (applied)
- Added an optional “apply embedded ICC and convert to sRGB” setting and implemented ICC conversion in the loader.

---

## Known issues / remaining work (current)

### A) Compare slider can still peg CPU on very large images
Even with throttling, wipe rendering on huge frames can be expensive.
- Next steps (if prioritized):
  - Ensure wipe rendering does not allocate new full-size pixmaps per tick.
  - Further reduce work on slider changes (e.g., repaint viewer only; avoid histogram work entirely while dragging).

### B) Auto Tone responsiveness is improved but not perfect
Auto Tone computation was moved off the UI thread, but users can still experience a noticeable CPU spike and temporary unresponsiveness during apply/re-render.
- Next steps (if prioritized):
  - Apply computed params without triggering redundant processing requests.
  - Consider a lower-res preview during Auto Tone apply for large images.

### C) Documentation upkeep
- User docs and technical docs were updated to reflect:
  - new compare slider
  - batch output quality controls
  - optional ICC behavior

---

## Verification checklist

### Quick code sanity
```bash
python -m compileall -q negative_converter
```

### Smoke test
1. Launch app.
2. Load a large image.
3. Use Auto Tone: confirm it completes and UI recovers quickly.
4. Drag Compare slider:
   - confirm it’s responsive and doesn’t lock up when moved to endpoints (0 / 100).

---

## Non-goals (for now)
- Full rewrite/refactor of architecture (keep changes incremental).
- Full RAW pipeline or full EXIF preservation unless explicitly requested.
