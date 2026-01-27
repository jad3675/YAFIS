// Combined adjustments shader - brightness, contrast, saturation, temp/tint
// Processes multiple adjustments in a single pass for efficiency

struct AdjustmentUniforms {
    brightness: f32,      // -100 to 100, converted to offset
    contrast: f32,        // contrast factor
    saturation: f32,      // saturation factor (1.0 = no change)
    temp: f32,            // temperature adjustment
    tint: f32,            // tint adjustment
    gamma: f32,           // gamma correction (1.0 = no change)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: AdjustmentUniforms;

// Convert RGB to HSV
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r / 255.0;
    let g = rgb.g / 255.0;
    let b = rgb.b / 255.0;
    
    let max_c = max(max(r, g), b);
    let min_c = min(min(r, g), b);
    let delta = max_c - min_c;
    
    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v = max_c;
    
    if (delta > 0.0001) {
        s = delta / max_c;
        
        if (max_c == r) {
            h = (g - b) / delta;
            if (g < b) {
                h += 6.0;
            }
        } else if (max_c == g) {
            h = 2.0 + (b - r) / delta;
        } else {
            h = 4.0 + (r - g) / delta;
        }
        h /= 6.0;
    }
    
    return vec3<f32>(h * 180.0, s * 255.0, v * 255.0);
}

// Convert HSV to RGB
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x / 180.0;
    let s = hsv.y / 255.0;
    let v = hsv.z / 255.0;
    
    if (s < 0.0001) {
        return vec3<f32>(v * 255.0, v * 255.0, v * 255.0);
    }
    
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    
    var r: f32;
    var g: f32;
    var b: f32;
    
    let sector = i32(i) % 6;
    if (sector == 0) {
        r = v; g = t; b = p;
    } else if (sector == 1) {
        r = q; g = v; b = p;
    } else if (sector == 2) {
        r = p; g = v; b = t;
    } else if (sector == 3) {
        r = p; g = q; b = v;
    } else if (sector == 4) {
        r = t; g = p; b = v;
    } else {
        r = v; g = p; b = q;
    }
    
    return vec3<f32>(r * 255.0, g * 255.0, b * 255.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;
    
    // 1. Brightness (additive)
    if (params.brightness != 0.0) {
        let offset = (params.brightness / 100.0) * 127.0;
        color = color + vec3<f32>(offset);
    }
    
    // 2. Contrast (multiplicative around midpoint)
    if (params.contrast != 1.0) {
        let midpoint = 128.0;
        color = (color - midpoint) * params.contrast + midpoint;
    }
    
    // 3. Temperature and Tint
    if (params.temp != 0.0 || params.tint != 0.0) {
        let temp_factor = params.temp * 0.3;
        let tint_factor = params.tint * 0.3;
        color.r = color.r + temp_factor;
        color.b = color.b - temp_factor;
        color.g = color.g - tint_factor;
    }
    
    // 4. Saturation (in HSV space)
    if (params.saturation != 1.0) {
        var hsv = rgb_to_hsv(color);
        hsv.y = hsv.y * params.saturation;
        hsv.y = clamp(hsv.y, 0.0, 255.0);
        color = hsv_to_rgb(hsv);
    }
    
    // 5. Gamma correction
    if (params.gamma != 1.0) {
        let normalized = color / 255.0;
        let gamma_corrected = pow(max(normalized, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma));
        color = gamma_corrected * 255.0;
    }
    
    // Clamp final result
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0));
    
    textureStore(output_tex, coords, vec4<f32>(color, 1.0));
}
