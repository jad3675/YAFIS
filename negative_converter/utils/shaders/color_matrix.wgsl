// Color Matrix shader - applies 3x3 color correction matrix
// Used for film profile color correction

struct MatrixUniforms {
    row0: vec4<f32>,  // [m00, m01, m02, 0]
    row1: vec4<f32>,  // [m10, m11, m12, 0]
    row2: vec4<f32>,  // [m20, m21, m22, 0]
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: MatrixUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0);
    
    // Apply 3x3 matrix multiplication
    let rgb = color.rgb;
    let corrected = vec3<f32>(
        dot(rgb, params.row0.xyz),
        dot(rgb, params.row1.xyz),
        dot(rgb, params.row2.xyz)
    );
    
    textureStore(output_tex, coords, vec4<f32>(corrected, 1.0));
}
