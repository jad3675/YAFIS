// Invert shader - converts negative to positive
// Input: uint8 image normalized to 0.0-1.0 range (multiply by 255 before upload)
// Output: inverted image in 0.0-255.0 range

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0);
    
    // Invert: 255 - value (input is in 0-255 range as float)
    let inverted = vec3<f32>(
        255.0 - color.r,
        255.0 - color.g,
        255.0 - color.b
    );
    
    textureStore(output_tex, coords, vec4<f32>(inverted, 1.0));
}
