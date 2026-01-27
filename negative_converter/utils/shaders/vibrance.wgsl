// Vibrance adjustment shader
// Boosts less saturated colors more than already saturated ones

struct VibranceUniforms {
    vibrance: f32,    // Vibrance amount (-100 to 100)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: VibranceUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // Normalize to 0-1
    let rgb = color / 255.0;
    
    // Calculate current saturation (simple approximation)
    let max_c = max(max(rgb.r, rgb.g), rgb.b);
    let min_c = min(min(rgb.r, rgb.g), rgb.b);
    let current_sat = select(0.0, (max_c - min_c) / max_c, max_c > 0.0);
    
    // Vibrance factor - boost less saturated colors more
    let factor = params.vibrance / 100.0;
    let boost = factor * (1.0 - current_sat);
    
    // Calculate luminance for saturation adjustment
    let lum = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    
    // Apply vibrance as saturation boost
    var result = rgb + (rgb - vec3<f32>(lum)) * boost;
    
    result = clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0;
    textureStore(output_tex, coords, vec4<f32>(result, 1.0));
}
