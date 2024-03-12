struct AuxConstantParameters {
    grid_size_z: i32,
    grid_size_x: i32,
    dz: i32,
    dx: i32,
    dt: i32,
    source_z: i32,
    source_x: i32,
};

// Group 0 - Parameters

@group(0) @binding(0)
var<storage,read_write> auxcp: AuxConstantParameters;

@group(0) @binding(6) // source term
var<storage,read> source: array<f32>;

// Group 1 - Simulation Arrays

@group(1) @binding(1) // pressure field present
var<storage,read_write> P_present: array<f32>;

@group(1) @binding(2) // pressure field past
var<storage,read_write> P_past: array<f32>;

@group(1) @binding(3) // pressure field future
var<storage,read_write> P_future: array<f32>;

@group(1) @binding(4) // laplacian matrix
var<storage,read_write> lap: array<f32>;

@group(1) @binding(5) // velocity map
var<storage,read> c: array<f32>;

var<workgroup> pzz: array<f32, 2500>;
var<workgroup> pxx: array<f32, 2500>;

@compute
@workgroup_size(wsx, wsy)
fn laplacian_5_operator(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    pzz[x + z * auxcp.grid_size_x] = f32(0);
    pxx[x + z * auxcp.grid_size_x] = f32(0);

    if (z >= 2 && z <= 47)
    {
        pzz[x + z * auxcp.grid_size_x] = f32((f32(-1)/f32(12)) * f32(P_present[(z + 2) * auxcp.grid_size_x + x]) + (f32(4)/f32(3)) * f32(P_present[(z + 1) * auxcp.grid_size_x + x]) - (f32(5)/f32(2)) * f32(P_present[x + z * auxcp.grid_size_x]) + (f32(4)/f32(3)) * f32(P_present[(z - 1) * auxcp.grid_size_x + x]) - (f32(1)/f32(12)) * f32(P_present[(z - 2) * auxcp.grid_size_x + x])) / f32(1);
    }
    if (x >= 2 && x <= 47)
    {
        pxx[x + z * auxcp.grid_size_x] = f32((f32(-1)/f32(12)) * f32(P_present[z * auxcp.grid_size_x + (x + 2)]) + (f32(4)/f32(3)) * f32(P_present[z * auxcp.grid_size_x + (x + 1)]) - (f32(5)/f32(2)) * f32(P_present[x + z * auxcp.grid_size_x]) + (f32(4)/f32(3)) * f32(P_present[z * auxcp.grid_size_x + (x - 1)]) - (f32(1)/f32(12)) * f32(P_present[z * auxcp.grid_size_x + (x - 2)])) / f32(1);
    }

    lap[x + z * auxcp.grid_size_x] = pzz[x + z * auxcp.grid_size_x] + pxx[x + z * auxcp.grid_size_x];
}

@compute
@workgroup_size(wsx, wsy)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    P_future[x + z * auxcp.grid_size_x] = (c[x + z * auxcp.grid_size_x] * c[x + z * auxcp.grid_size_x]) * lap[x + z * auxcp.grid_size_x] * (0.001 * 0.001);

    P_future[x + z * auxcp.grid_size_x] += ((f32(2) * P_present[x + z * auxcp.grid_size_x]) - P_past[x + z * auxcp.grid_size_x]);

    P_past[x + z * auxcp.grid_size_x] = P_present[x + z * auxcp.grid_size_x];
    P_present[x + z * auxcp.grid_size_x] = P_future[x + z * auxcp.grid_size_x];

    if (z == 49 && x == 49)
    {
        P_future[auxcp.source_x + auxcp.source_z * auxcp.grid_size_x] = f32(5000);
    }
    if (z == 49 && x == 47)
    {
        P_future[auxcp.source_x + auxcp.source_z * auxcp.grid_size_x] = f32(0);
    }
}
