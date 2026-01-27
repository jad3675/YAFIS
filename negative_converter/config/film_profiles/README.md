# Film Profiles

This directory contains JSON configuration files that define processing parameters for different film types.

## Overview

Film profiles customize how negatives are converted to positives. Each profile can adjust:
- Color correction matrix
- Per-channel gamma curves
- Saturation boost
- LAB color correction

## Built-in Profiles

| Profile | File | Description |
|---------|------|-------------|
| C-41 | `c41_generic.json` | Standard color negative film (Kodak Portra, Fuji 400H, etc.) |
| ECN-2 | `ecn2_generic.json` | Motion picture color negative (Kodak Vision3, etc.) |
| E-6 | `e6_generic.json` | Slide/reversal film (Kodak Ektachrome, Fuji Velvia, etc.) |
| B&W | `bw_generic.json` | Black and white negative film |

## Profile Structure

```json
{
    "id": "profile_id",
    "name": "Human Readable Name",
    "description": "Description of the profile",
    "correction_matrix": [
        [r_to_r, g_to_r, b_to_r],
        [r_to_g, g_to_g, b_to_g],
        [r_to_b, g_to_b, b_to_b]
    ],
    "gamma": {
        "red": 1.0,
        "green": 1.0,
        "blue": 1.0
    },
    "saturation_boost": 1.0,
    "lab_correction": {
        "a_target": 128.0,
        "a_factor": 0.5,
        "b_target": 128.0,
        "b_factor": 0.5
    }
}
```

## Parameter Reference

### correction_matrix
A 3x3 color correction matrix applied after inversion and white balance.
- Values > 1.0 on diagonal increase that channel's intensity
- Off-diagonal values control color mixing/cross-talk
- Typical range: -0.5 to 2.0

**Example - Boost red, reduce blue cross-talk:**
```json
"correction_matrix": [
    [1.6, -0.2, -0.1],
    [-0.1, 1.5, -0.1],
    [-0.1, -0.3, 1.4]
]
```

### gamma
Per-channel gamma correction applied during curve processing.
- Values < 1.0 brighten midtones
- Values > 1.0 darken midtones
- Typical range: 0.8 to 1.2

**Example - Warm up shadows (common for C-41):**
```json
"gamma": {
    "red": 0.95,
    "green": 1.0,
    "blue": 1.1
}
```

### saturation_boost
Multiplier for saturation in HSV color space.
- 1.0 = no change
- > 1.0 = more saturated
- < 1.0 = less saturated (use 0.0 for B&W)
- Typical range: 0.0 to 1.5

### lab_correction
Adjusts color balance in LAB color space.

- **a_target**: Target value for A channel (green-magenta axis). 128 = neutral.
- **a_factor**: How aggressively to correct toward target (0.0-1.0)
- **b_target**: Target value for B channel (blue-yellow axis). 128 = neutral.
- **b_factor**: How aggressively to correct toward target (0.0-1.0)

## Creating Custom Profiles

1. Copy an existing profile as a starting point
2. Rename with a descriptive filename (e.g., `portra-400.json`)
3. Update the `id` and `name` fields
4. Adjust parameters to taste
5. Place in this directory

**Tips:**
- Start with small adjustments (±0.1 for gamma, ±0.2 for matrix)
- Test with multiple images from the same film stock
- The correction matrix has the biggest impact on color rendition
- Use LAB correction for fine-tuning color casts

## Film-Specific Notes

### C-41 (Color Negative)
- Has orange mask that must be neutralized
- Typically needs matrix values > 1.0 on diagonal
- Often benefits from slight red gamma reduction (warmer shadows)

### ECN-2 (Motion Picture)
- Darker orange/brown mask than C-41
- May need stronger correction matrix
- Often has more contrast built into the film

### E-6 (Slide/Reversal)
- No orange mask (clear base)
- Minimal color correction needed
- Higher saturation boost often desired

### B&W
- No color correction needed (identity matrix)
- Set saturation_boost to 0.0
- LAB correction factors should be 0.0
