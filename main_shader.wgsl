struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    source_z: i32,
    source_x: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
    dt: f32,
};

// Group 0 - Constants Parameters

@group(0) @binding(0) // Info Int
var<storage,read> infoI32: InfoInt;

@group(0) @binding(1) // Info Float
var<storage,read> infoF32: InfoFloat;

@group(0) @binding(2) // source term
var<storage,read> source: array<f32>;

@group(0) @binding(3) // velocity map
var<storage,read> c: array<f32>;

// Group 1 - Simulation Arrays

@group(1) @binding(4) // pressure field present
var<storage,read_write> P_present: array<f32>;

@group(1) @binding(5) // pressure field past
var<storage,read_write> P_past: array<f32>;

@group(1) @binding(6) // pressure field future
var<storage,read_write> P_future: array<f32>;

@group(1) @binding(7) // laplacian matrix
var<storage,read_write> lap: array<f32>;

@compute
@workgroup_size(wsx, wsy)
fn laplacian_5_operator(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pzz: f32 = f32(0);
    var pxx: f32 = f32(0);

    if (z >= 2 && z <= infoI32.grid_size_z - 3)
    {
        pzz = f32((f32(-1)/f32(12)) * f32(P_present[(z + 2) * infoI32.grid_size_x + x]) + (f32(4)/f32(3)) * f32(P_present[(z + 1) * infoI32.grid_size_x + x]) - (f32(5)/f32(2)) * f32(P_present[x + z * infoI32.grid_size_x]) + (f32(4)/f32(3)) * f32(P_present[(z - 1) * infoI32.grid_size_x + x]) - (f32(1)/f32(12)) * f32(P_present[(z - 2) * infoI32.grid_size_x + x])) / (infoF32.dz * infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3)
    {
        pxx = f32((f32(-1)/f32(12)) * f32(P_present[z * infoI32.grid_size_x + (x + 2)]) + (f32(4)/f32(3)) * f32(P_present[z * infoI32.grid_size_x + (x + 1)]) - (f32(5)/f32(2)) * f32(P_present[x + z * infoI32.grid_size_x]) + (f32(4)/f32(3)) * f32(P_present[z * infoI32.grid_size_x + (x - 1)]) - (f32(1)/f32(12)) * f32(P_present[z * infoI32.grid_size_x + (x - 2)])) / (infoF32.dx * infoF32.dx);
    }

    lap[x + z * infoI32.grid_size_x] = pzz + pxx;
}

@compute
@workgroup_size(wsx, wsy)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    P_future[x + z * infoI32.grid_size_x] = (c[x + z * infoI32.grid_size_x] * c[x + z * infoI32.grid_size_x]) * lap[x + z * infoI32.grid_size_x] * (infoF32.dt * infoF32.dt);

    P_future[x + z * infoI32.grid_size_x] += ((f32(2) * P_present[x + z * infoI32.grid_size_x]) - P_past[x + z * infoI32.grid_size_x]);

    if (z == infoI32.source_z && x == infoI32.source_x)
    {
        P_future[x + z * infoI32.grid_size_x] += f32(5000);
    }

    P_past[x + z * infoI32.grid_size_x] = P_present[x + z * infoI32.grid_size_x];
    P_present[x + z * infoI32.grid_size_x] = P_future[x + z * infoI32.grid_size_x];
}
