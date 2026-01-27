// Shadows and Highlights adjustment shader
// Works in LAB-like luminance space for natural results

struct SHUniforms {
    shadows: f32,      // Shadow adjustment (-100 to 100)
    highlights: f32,   // Highlight adjustment (-100 to 100)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: SHUniforms;

// Approximate RGB to luminance
fn get_luminance(rgb: vec3<f32>) -> f32 {
    return 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // Get normalized luminance (0-1)
    let lum = get_luminance(color) / 255.0;
    
    // Calculate shadow and highlight masks with smooth falloff
    let shadow_mask = pow(clamp(1.0 - lum, 0.0, 1.0), 1.5);
    let highlight_mask = pow(clamp(lum, 0.0, 1.0), 1.5);
    
    // Calculate adjustments
    let shadow_adjust = (params.shadows / 100.0) * 100.0;
    let highlight_adjust = (params.highlights / 100.0) * 100.0;
    
    // Apply adjustments
    let adjustment = shadow_mask * shadow_adjust + highlight_mask * highlight_adjust;
    var result = color + vec3<f32>(adjustment);
    
    result = clamp(result, vec3<f32>(0.0), vec3<f32>(255.0));
    textureStore(output_tex, coords, vec4<f32>(result, 1.0));
}
