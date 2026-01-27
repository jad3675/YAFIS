// Finalize shader - clamps values and prepares for output
// Converts from float32 processing range to uint8 output range

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // Clamp to valid range and round to nearest integer
    let result = clamp(round(color), vec3<f32>(0.0), vec3<f32>(255.0));
    
    textureStore(output_tex, coords, vec4<f32>(result, 1.0));
}
