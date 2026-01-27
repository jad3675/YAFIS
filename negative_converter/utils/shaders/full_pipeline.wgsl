// Full Pipeline Shader - All processing stages in one dispatch
// Extended version with Selective Color, LAB grading, and edge-aware smoothing

struct PipelineUniforms {
    // Stage flags (which operations to apply)
    do_invert: u32,           // 0 or 1
    do_white_balance: u32,    // 0 or 1
    do_color_matrix: u32,     // 0 or 1
    do_adjustments: u32,      // 0 or 1
    
    // White balance
    wb_scale_r: f32,
    wb_scale_g: f32,
    wb_scale_b: f32,
    _pad0: f32,
    
    // Color matrix (3x3, row-major, padded)
    matrix_row0: vec4<f32>,
    matrix_row1: vec4<f32>,
    matrix_row2: vec4<f32>,
    
    // Adjustments
    brightness: f32,
    contrast: f32,
    saturation: f32,
    temp: f32,
    tint: f32,
    gamma: f32,
    shadows: f32,
    highlights: f32,
    vibrance: f32,
    hue_shift: f32,
    _pad2: f32,
    _pad3: f32,
    
    // Levels
    do_levels: u32,
    in_black: f32,
    in_white: f32,
    lvl_gamma: f32,
    out_black: f32,
    out_white: f32,
    _pad4: f32,
    _pad5: f32,
    
    // Channel Mixer
    do_channel_mixer: u32,
    mixer_row0: vec4<f32>,
    mixer_row1: vec4<f32>,
    mixer_row2: vec4<f32>,
    
    // Curves
    do_curves: u32,
    _pad6: f32,
    _pad7: f32,
    _pad8: f32,
    
    // HSL adjustments
    do_hsl: u32,
    hsl_reds: vec4<f32>,
    hsl_yellows: vec4<f32>,
    hsl_greens: vec4<f32>,
    hsl_cyans: vec4<f32>,
    hsl_blues: vec4<f32>,
    hsl_magentas: vec4<f32>,

    // Selective Color (CMYK adjustments per color range)
    do_selective_color: u32,
    sel_relative: u32,  // 0 = absolute, 1 = relative
    // Each vec4: [cyan, magenta, yellow, black] adjustments (-100 to 100)
    sel_reds: vec4<f32>,
    sel_yellows: vec4<f32>,
    sel_greens: vec4<f32>,
    sel_cyans: vec4<f32>,
    sel_blues: vec4<f32>,
    sel_magentas: vec4<f32>,
    sel_whites: vec4<f32>,
    sel_neutrals: vec4<f32>,
    sel_blacks: vec4<f32>,
    
    // LAB color grading
    do_lab_grading: u32,
    lab_l_shift: f32,      // Lightness shift
    lab_a_shift: f32,      // a* shift (green-red)
    lab_b_shift: f32,      // b* shift (blue-yellow)
    lab_a_target: f32,     // Target a* for correction
    lab_a_factor: f32,     // Correction factor for a*
    lab_b_target: f32,     // Target b* for correction
    lab_b_factor: f32,     // Correction factor for b*
    
    // Edge-aware smoothing (simplified bilateral)
    do_smoothing: u32,
    smooth_radius: f32,    // Kernel radius (1-5)
    smooth_sigma_s: f32,   // Spatial sigma
    smooth_sigma_r: f32,   // Range (color) sigma
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: PipelineUniforms;
@group(0) @binding(3) var curves_lut: texture_2d<f32>;

const PI: f32 = 3.14159265359;

// =========================================================================
// Color Space Conversion Functions
// =========================================================================

fn get_luminance(rgb: vec3<f32>) -> f32 {
    return 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
}

// RGB to HSV (H: 0-360, S: 0-1, V: 0-1)
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r / 255.0;
    let g = rgb.g / 255.0;
    let b = rgb.b / 255.0;
    
    let max_c = max(max(r, g), b);
    let min_c = min(min(r, g), b);
    let delta = max_c - min_c;
    
    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v: f32 = max_c;
    
    if (delta > 0.00001) {
        s = delta / max_c;
        if (max_c == r) {
            h = 60.0 * (((g - b) / delta) % 6.0);
        } else if (max_c == g) {
            h = 60.0 * (((b - r) / delta) + 2.0);
        } else {
            h = 60.0 * (((r - g) / delta) + 4.0);
        }
        if (h < 0.0) { h = h + 360.0; }
    }
    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
    
    let c = v * s;
    let x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    if (h < 60.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (h < 120.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (h < 180.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (h < 240.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (h < 300.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    
    return (rgb + vec3<f32>(m)) * 255.0;
}

// RGB to HLS (H: 0-360, L: 0-1, S: 0-1)
fn rgb_to_hls(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r / 255.0;
    let g = rgb.g / 255.0;
    let b = rgb.b / 255.0;
    
    let max_c = max(max(r, g), b);
    let min_c = min(min(r, g), b);
    let delta = max_c - min_c;
    let l = (max_c + min_c) / 2.0;
    
    var h: f32 = 0.0;
    var s: f32 = 0.0;
    
    if (delta > 0.00001) {
        s = select(delta / (max_c + min_c), delta / (2.0 - max_c - min_c), l >= 0.5);
        if (max_c == r) { h = 60.0 * (((g - b) / delta) % 6.0); }
        else if (max_c == g) { h = 60.0 * (((b - r) / delta) + 2.0); }
        else { h = 60.0 * (((r - g) / delta) + 4.0); }
        if (h < 0.0) { h = h + 360.0; }
    }
    return vec3<f32>(h, l, s);
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    var tt = t;
    if (tt < 0.0) { tt = tt + 1.0; }
    if (tt > 1.0) { tt = tt - 1.0; }
    if (tt < 1.0/6.0) { return p + (q - p) * 6.0 * tt; }
    if (tt < 0.5) { return q; }
    if (tt < 2.0/3.0) { return p + (q - p) * (2.0/3.0 - tt) * 6.0; }
    return p;
}

fn hls_to_rgb(hls: vec3<f32>) -> vec3<f32> {
    let h = hls.x;
    let l = hls.y;
    let s = hls.z;
    
    if (s < 0.00001) { return vec3<f32>(l * 255.0); }
    
    let q = select(l * (1.0 + s), l + s - l * s, l >= 0.5);
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;
    
    let r = hue_to_rgb(p, q, h_norm + 1.0/3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0/3.0);
    
    return vec3<f32>(r, g, b) * 255.0;
}

// RGB to LAB (approximate, using D65 illuminant)
fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    // Normalize to 0-1
    var r = rgb.r / 255.0;
    var g = rgb.g / 255.0;
    var b = rgb.b / 255.0;
    
    // sRGB to linear
    r = select(r / 12.92, pow((r + 0.055) / 1.055, 2.4), r > 0.04045);
    g = select(g / 12.92, pow((g + 0.055) / 1.055, 2.4), g > 0.04045);
    b = select(b / 12.92, pow((b + 0.055) / 1.055, 2.4), b > 0.04045);
    
    // RGB to XYZ (D65)
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
    
    // Normalize by D65 white point
    let xn = x / 0.95047;
    let yn = y / 1.00000;
    let zn = z / 1.08883;
    
    // XYZ to LAB
    let epsilon = 0.008856;
    let kappa = 903.3;
    
    let fx = select(pow(xn, 1.0/3.0), (kappa * xn + 16.0) / 116.0, xn > epsilon);
    let fy = select(pow(yn, 1.0/3.0), (kappa * yn + 16.0) / 116.0, yn > epsilon);
    let fz = select(pow(zn, 1.0/3.0), (kappa * zn + 16.0) / 116.0, zn > epsilon);
    
    let L = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);
    
    return vec3<f32>(L, a + 128.0, bb + 128.0);  // Shift a,b to 0-255 range
}

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    let L = lab.x;
    let a = lab.y - 128.0;  // Shift back
    let bb = lab.z - 128.0;
    
    // LAB to XYZ
    let fy = (L + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - bb / 200.0;
    
    let epsilon = 0.008856;
    let kappa = 903.3;
    
    let xr = select(fx * fx * fx, (116.0 * fx - 16.0) / kappa, fx * fx * fx > epsilon);
    let yr = select(fy * fy * fy, (116.0 * fy - 16.0) / kappa, L > kappa * epsilon);
    let zr = select(fz * fz * fz, (116.0 * fz - 16.0) / kappa, fz * fz * fz > epsilon);
    
    // Denormalize by D65 white point
    let x = xr * 0.95047;
    let y = yr * 1.00000;
    let z = zr * 1.08883;
    
    // XYZ to linear RGB
    var r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314;
    var g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560;
    var b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252;
    
    // Linear to sRGB
    r = select(r * 12.92, 1.055 * pow(r, 1.0/2.4) - 0.055, r > 0.0031308);
    g = select(g * 12.92, 1.055 * pow(g, 1.0/2.4) - 0.055, g > 0.0031308);
    b = select(b * 12.92, 1.055 * pow(b, 1.0/2.4) - 0.055, b > 0.0031308);
    
    return clamp(vec3<f32>(r, g, b) * 255.0, vec3<f32>(0.0), vec3<f32>(255.0));
}

// Get HSL adjustment for a given hue
fn get_hsl_adjustment(h: f32) -> vec3<f32> {
    var adj = vec3<f32>(0.0);
    
    if (h < 30.0 || h >= 330.0) {
        let weight = select(1.0 - h / 30.0, 1.0 - (360.0 - h) / 30.0, h >= 330.0);
        adj = adj + params.hsl_reds.xyz * clamp(1.0 - abs(weight), 0.0, 1.0);
    }
    if (h >= 15.0 && h < 75.0) {
        let weight = 1.0 - abs(h - 45.0) / 30.0;
        adj = adj + params.hsl_yellows.xyz * clamp(weight, 0.0, 1.0);
    }
    if (h >= 60.0 && h < 150.0) {
        let weight = 1.0 - abs(h - 105.0) / 45.0;
        adj = adj + params.hsl_greens.xyz * clamp(weight, 0.0, 1.0);
    }
    if (h >= 135.0 && h < 195.0) {
        let weight = 1.0 - abs(h - 165.0) / 30.0;
        adj = adj + params.hsl_cyans.xyz * clamp(weight, 0.0, 1.0);
    }
    if (h >= 180.0 && h < 270.0) {
        let weight = 1.0 - abs(h - 225.0) / 45.0;
        adj = adj + params.hsl_blues.xyz * clamp(weight, 0.0, 1.0);
    }
    if (h >= 255.0 && h < 345.0) {
        let weight = 1.0 - abs(h - 300.0) / 45.0;
        adj = adj + params.hsl_magentas.xyz * clamp(weight, 0.0, 1.0);
    }
    return adj;
}

// Get selective color adjustment for a pixel
fn get_selective_color_adjustment(rgb: vec3<f32>) -> vec4<f32> {
    let hsv = rgb_to_hsv(rgb);
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
    
    var adj = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    // Color ranges based on hue
    // Reds: 0-30, 330-360
    if (h < 30.0 || h >= 330.0) {
        let weight = select(1.0 - h / 30.0, 1.0 - (360.0 - h) / 30.0, h >= 330.0);
        let w = clamp(1.0 - abs(weight), 0.0, 1.0) * s;
        adj = adj + params.sel_reds * w;
        total_weight = total_weight + w;
    }
    // Yellows: 30-90
    if (h >= 15.0 && h < 75.0) {
        let w = clamp(1.0 - abs(h - 45.0) / 30.0, 0.0, 1.0) * s;
        adj = adj + params.sel_yellows * w;
        total_weight = total_weight + w;
    }
    // Greens: 90-150
    if (h >= 60.0 && h < 150.0) {
        let w = clamp(1.0 - abs(h - 105.0) / 45.0, 0.0, 1.0) * s;
        adj = adj + params.sel_greens * w;
        total_weight = total_weight + w;
    }
    // Cyans: 150-210
    if (h >= 135.0 && h < 195.0) {
        let w = clamp(1.0 - abs(h - 165.0) / 30.0, 0.0, 1.0) * s;
        adj = adj + params.sel_cyans * w;
        total_weight = total_weight + w;
    }
    // Blues: 210-270
    if (h >= 180.0 && h < 270.0) {
        let w = clamp(1.0 - abs(h - 225.0) / 45.0, 0.0, 1.0) * s;
        adj = adj + params.sel_blues * w;
        total_weight = total_weight + w;
    }
    // Magentas: 270-330
    if (h >= 255.0 && h < 345.0) {
        let w = clamp(1.0 - abs(h - 300.0) / 45.0, 0.0, 1.0) * s;
        adj = adj + params.sel_magentas * w;
        total_weight = total_weight + w;
    }
    
    // Grayscale ranges based on value and saturation
    let sat_thresh = 0.15;
    if (s < sat_thresh) {
        let gray_weight = 1.0 - s / sat_thresh;
        // Whites: high value
        if (v > 0.85) {
            let w = gray_weight * (v - 0.85) / 0.15;
            adj = adj + params.sel_whites * w;
            total_weight = total_weight + w;
        }
        // Blacks: low value
        if (v < 0.15) {
            let w = gray_weight * (0.15 - v) / 0.15;
            adj = adj + params.sel_blacks * w;
            total_weight = total_weight + w;
        }
        // Neutrals: mid value
        if (v >= 0.15 && v <= 0.85) {
            let w = gray_weight * (1.0 - abs(v - 0.5) / 0.35);
            adj = adj + params.sel_neutrals * w;
            total_weight = total_weight + w;
        }
    }
    
    if (total_weight > 0.0) {
        adj = adj / total_weight;
    }
    return adj;
}

// Apply selective color CMYK adjustment
fn apply_selective_color(rgb: vec3<f32>, adj: vec4<f32>, relative: bool) -> vec3<f32> {
    // Convert RGB to CMY
    let r = rgb.r / 255.0;
    let g = rgb.g / 255.0;
    let b = rgb.b / 255.0;
    
    var c = 1.0 - r;
    var m = 1.0 - g;
    var y = 1.0 - b;
    
    // Calculate K (black)
    let k = min(min(c, m), y);
    
    // Remove K from CMY
    c = c - k;
    m = m - k;
    y = y - k;
    
    // Apply adjustments (adj: cyan, magenta, yellow, black)
    let c_adj = adj.x / 100.0;
    let m_adj = adj.y / 100.0;
    let y_adj = adj.z / 100.0;
    let k_adj = adj.w / 100.0;
    
    if (relative) {
        c = c * (1.0 + c_adj);
        m = m * (1.0 + m_adj);
        y = y * (1.0 + y_adj);
    } else {
        c = c + c_adj;
        m = m + m_adj;
        y = y + y_adj;
    }
    
    // Adjust K
    var k_new = k * (1.0 - k_adj);
    
    // Clamp
    c = clamp(c, 0.0, 1.0 - k_new);
    m = clamp(m, 0.0, 1.0 - k_new);
    y = clamp(y, 0.0, 1.0 - k_new);
    k_new = clamp(k_new, 0.0, 1.0);
    
    // Convert back to RGB
    let r_out = (1.0 - c - k_new);
    let g_out = (1.0 - m - k_new);
    let b_out = (1.0 - y - k_new);
    
    return clamp(vec3<f32>(r_out, g_out, b_out) * 255.0, vec3<f32>(0.0), vec3<f32>(255.0));
}

// Gaussian weight for bilateral filter
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;
    
    // =========================================================================
    // Stage 1: Invert
    // =========================================================================
    if (params.do_invert == 1u) {
        color = vec3<f32>(255.0) - color;
    }
    
    // =========================================================================
    // Stage 2: White Balance
    // =========================================================================
    if (params.do_white_balance == 1u) {
        color = color * vec3<f32>(params.wb_scale_r, params.wb_scale_g, params.wb_scale_b);
    }
    
    // =========================================================================
    // Stage 3: Color Matrix
    // =========================================================================
    if (params.do_color_matrix == 1u) {
        let rgb = color;
        color = vec3<f32>(
            dot(rgb, params.matrix_row0.xyz),
            dot(rgb, params.matrix_row1.xyz),
            dot(rgb, params.matrix_row2.xyz)
        );
    }
    
    // =========================================================================
    // Stage 4: Basic Adjustments
    // =========================================================================
    if (params.do_adjustments == 1u) {
        if (params.brightness != 0.0) {
            color = color + vec3<f32>((params.brightness / 100.0) * 127.0);
        }
        if (params.contrast != 1.0) {
            color = (color - 128.0) * params.contrast + 128.0;
        }
        if (params.temp != 0.0 || params.tint != 0.0) {
            color.r = color.r + params.temp * 0.3;
            color.b = color.b - params.temp * 0.3;
            color.g = color.g - params.tint * 0.3;
        }
        if (params.gamma != 1.0) {
            let normalized = color / 255.0;
            color = pow(max(normalized, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma)) * 255.0;
        }
        if (params.shadows != 0.0 || params.highlights != 0.0) {
            let lum = get_luminance(color) / 255.0;
            let shadow_mask = pow(clamp(1.0 - lum, 0.0, 1.0), 1.5);
            let highlight_mask = pow(clamp(lum, 0.0, 1.0), 1.5);
            let adjustment = shadow_mask * params.shadows + highlight_mask * params.highlights;
            color = color + vec3<f32>(adjustment);
        }
        if (params.vibrance != 0.0) {
            let rgb = color / 255.0;
            let max_c = max(max(rgb.r, rgb.g), rgb.b);
            let min_c = min(min(rgb.r, rgb.g), rgb.b);
            let current_sat = select(0.0, (max_c - min_c) / max_c, max_c > 0.0);
            let boost = (params.vibrance / 100.0) * (1.0 - current_sat);
            let lum = get_luminance(rgb);
            color = (rgb + (rgb - vec3<f32>(lum)) * boost) * 255.0;
        }
        if (params.saturation != 1.0) {
            let rgb = color / 255.0;
            let lum = get_luminance(rgb);
            color = (vec3<f32>(lum) + (rgb - vec3<f32>(lum)) * params.saturation) * 255.0;
        }
        if (params.hue_shift != 0.0) {
            var hsv = rgb_to_hsv(clamp(color, vec3<f32>(0.0), vec3<f32>(255.0)));
            hsv.x = (hsv.x + params.hue_shift + 360.0) % 360.0;
            color = hsv_to_rgb(hsv);
        }
    }

    // =========================================================================
    // Stage 5: Levels
    // =========================================================================
    if (params.do_levels == 1u) {
        let in_range = max(params.in_white - params.in_black, 0.001);
        let out_range = params.out_white - params.out_black;
        for (var c = 0; c < 3; c++) {
            var val = color[c];
            val = (val - params.in_black) / in_range;
            val = clamp(val, 0.0, 1.0);
            if (params.lvl_gamma != 1.0) {
                val = pow(val, 1.0 / params.lvl_gamma);
            }
            color[c] = val * out_range + params.out_black;
        }
    }
    
    // =========================================================================
    // Stage 6: Channel Mixer
    // =========================================================================
    if (params.do_channel_mixer == 1u) {
        let rgb = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0));
        let r_out = dot(rgb, params.mixer_row0.xyz / 100.0) + params.mixer_row0.w * 1.275;
        let g_out = dot(rgb, params.mixer_row1.xyz / 100.0) + params.mixer_row1.w * 1.275;
        let b_out = dot(rgb, params.mixer_row2.xyz / 100.0) + params.mixer_row2.w * 1.275;
        color = vec3<f32>(r_out, g_out, b_out);
    }
    
    // =========================================================================
    // Stage 7: Curves
    // =========================================================================
    if (params.do_curves == 1u) {
        let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0));
        let rgb_r = textureLoad(curves_lut, vec2<i32>(i32(clamped.r), 3), 0).r;
        let rgb_g = textureLoad(curves_lut, vec2<i32>(i32(clamped.g), 3), 0).r;
        let rgb_b = textureLoad(curves_lut, vec2<i32>(i32(clamped.b), 3), 0).r;
        let r_out = textureLoad(curves_lut, vec2<i32>(i32(rgb_r), 0), 0).r;
        let g_out = textureLoad(curves_lut, vec2<i32>(i32(rgb_g), 1), 0).r;
        let b_out = textureLoad(curves_lut, vec2<i32>(i32(rgb_b), 2), 0).r;
        color = vec3<f32>(r_out, g_out, b_out);
    }
    
    // =========================================================================
    // Stage 8: HSL Per-Range
    // =========================================================================
    if (params.do_hsl == 1u) {
        var hls = rgb_to_hls(clamp(color, vec3<f32>(0.0), vec3<f32>(255.0)));
        let adj = get_hsl_adjustment(hls.x);
        hls.x = (hls.x + adj.x + 360.0) % 360.0;
        hls.z = clamp(hls.z * (1.0 + adj.y / 100.0), 0.0, 1.0);
        hls.y = clamp(hls.y + adj.z / 200.0, 0.0, 1.0);
        color = hls_to_rgb(hls);
    }
    
    // =========================================================================
    // Stage 9: Selective Color
    // =========================================================================
    if (params.do_selective_color == 1u) {
        let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0));
        let adj = get_selective_color_adjustment(clamped);
        let is_relative = params.sel_relative == 1u;
        color = apply_selective_color(clamped, adj, is_relative);
    }
    
    // =========================================================================
    // Stage 10: LAB Color Grading
    // =========================================================================
    if (params.do_lab_grading == 1u) {
        var lab = rgb_to_lab(clamp(color, vec3<f32>(0.0), vec3<f32>(255.0)));
        
        // Apply direct shifts
        lab.x = clamp(lab.x + params.lab_l_shift, 0.0, 100.0);
        lab.y = clamp(lab.y + params.lab_a_shift, 0.0, 255.0);
        lab.z = clamp(lab.z + params.lab_b_shift, 0.0, 255.0);
        
        // Apply target-based correction (like in converter)
        if (params.lab_a_factor != 0.0) {
            let a_diff = lab.y - params.lab_a_target;
            lab.y = lab.y - a_diff * params.lab_a_factor;
        }
        if (params.lab_b_factor != 0.0) {
            let b_diff = lab.z - params.lab_b_target;
            lab.z = lab.z - b_diff * params.lab_b_factor;
        }
        
        color = lab_to_rgb(lab);
    }
    
    // =========================================================================
    // Stage 11: Edge-Aware Smoothing (Simplified Bilateral)
    // =========================================================================
    if (params.do_smoothing == 1u) {
        let radius = i32(params.smooth_radius);
        let sigma_s = params.smooth_sigma_s;
        let sigma_r = params.smooth_sigma_r;
        
        var sum_color = vec3<f32>(0.0);
        var sum_weight = 0.0;
        let center_color = color;
        
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dx = -radius; dx <= radius; dx++) {
                let sample_coords = coords + vec2<i32>(dx, dy);
                
                // Bounds check
                if (sample_coords.x >= 0 && sample_coords.x < i32(dims.x) &&
                    sample_coords.y >= 0 && sample_coords.y < i32(dims.y)) {
                    
                    let sample_color = textureLoad(input_tex, sample_coords, 0).rgb;
                    
                    // Spatial weight
                    let spatial_dist = sqrt(f32(dx * dx + dy * dy));
                    let spatial_weight = gaussian(spatial_dist, sigma_s);
                    
                    // Range (color) weight
                    let color_diff = length(sample_color - center_color);
                    let range_weight = gaussian(color_diff, sigma_r);
                    
                    let weight = spatial_weight * range_weight;
                    sum_color = sum_color + sample_color * weight;
                    sum_weight = sum_weight + weight;
                }
            }
        }
        
        if (sum_weight > 0.0) {
            color = sum_color / sum_weight;
        }
    }
    
    // =========================================================================
    // Final: Clamp output
    // =========================================================================
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0));
    textureStore(output_tex, coords, vec4<f32>(color, 1.0));
}
