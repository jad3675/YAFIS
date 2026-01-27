// Levels adjustment shader
// Applies input/output levels with gamma correction

struct LevelsUniforms {
    in_black: f32,    // Input black point (0-255)
    in_white: f32,    // Input white point (0-255)
    gamma: f32,       // Gamma correction
    out_black: f32,   // Output black point (0-255)
    out_white: f32,   // Output white point (0-255)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LevelsUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // Calculate input range
    let in_range = max(params.in_white - params.in_black, 0.001);
    let out_range = params.out_white - params.out_black;
    
    // Apply levels to each channel
    var result: vec3<f32>;
    for (var c = 0; c < 3; c++) {
        var val = color[c];
        
        // Map input range to 0-1
        val = (val - params.in_black) / in_range;
        val = clamp(val, 0.0, 1.0);
        
        // Apply gamma
        if (params.gamma != 1.0) {
            val = pow(val, 1.0 / params.gamma);
        }
        
        // Map to output range
        val = val * out_range + params.out_black;
        
        result[c] = val;
    }
    
    result = clamp(result, vec3<f32>(0.0), vec3<f32>(255.0));
    textureStore(output_tex, coords, vec4<f32>(result, 1.0));
}
