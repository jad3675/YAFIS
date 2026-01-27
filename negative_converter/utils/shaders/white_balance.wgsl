// White Balance shader - applies RGB scaling factors
// Used for mask neutralization and gray world AWB

struct WBUniforms {
    scale_r: f32,
    scale_g: f32,
    scale_b: f32,
    _pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: WBUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0);
    
    // Apply white balance scaling
    let balanced = vec3<f32>(
        color.r * params.scale_r,
        color.g * params.scale_g,
        color.b * params.scale_b
    );
    
    textureStore(output_tex, coords, vec4<f32>(balanced, 1.0));
}
