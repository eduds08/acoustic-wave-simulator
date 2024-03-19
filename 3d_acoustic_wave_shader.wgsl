struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    grid_size_y: i32,
    source_z: i32,
    source_x: i32,
    source_y: i32,
    i: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
    dy: f32,
    dt: f32,
};

// Group 0 - Constants Parameters

@group(0) @binding(0) // Info Int
var<storage,read_write> infoI32: InfoInt;

@group(0) @binding(1) // Info Float
var<storage,read> infoF32: InfoFloat;

@group(0) @binding(2) // source term
var<storage,read> source: array<f32>;

@group(0) @binding(3) // velocity map
var<storage,read> c: f32;

// Group 1 - Simulation Arrays

@group(1) @binding(4) // pressure field present
var<storage,read_write> P_present: array<f32>;

@group(1) @binding(5) // pressure field past
var<storage,read_write> P_past: array<f32>;

@group(1) @binding(6) // pressure field future
var<storage,read_write> P_future: array<f32>;

@group(1) @binding(7) // laplacian matrix
var<storage,read_write> lap: array<f32>;

fn zxy(z: i32, x: i32, y: i32) -> i32 {
    let index = x + y * infoI32.grid_size_x + z * infoI32.grid_size_x * infoI32.grid_size_y;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && y >= 0 && y < infoI32.grid_size_y && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx, wsy)
fn laplacian_5_operator(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let y: i32 = i32(index.z);

    var pzz: f32 = 0.;
    var pxx: f32 = 0.;
    var pyy: f32 = 0.;

    if (z >= 2 && z <= infoI32.grid_size_z - 3)
    {
        pzz = ((-1./12.) * P_present[zxy(z + 2, x, y)] + (4./3.) * P_present[zxy(z + 1, x, y)] - (5./2.) * P_present[zxy(z, x, y)] + (4./3.) * P_present[zxy(z - 1, x, y)] - (1./12.) * P_present[zxy(z - 2, x, y)]) / (infoF32.dz * infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3)
    {
        pxx = ((-1./12.) * P_present[zxy(z, x + 2, y)] + (4./3.) * P_present[zxy(z, x + 1, y)] - (5./2.) * P_present[zxy(z, x, y)] + (4./3.) * P_present[zxy(z, x - 1, y)] - (1./12.) * P_present[zxy(z, x - 2, y)]) / (infoF32.dx * infoF32.dx);
    }
    if (y >= 2 && y <= infoI32.grid_size_y - 3)
    {
        pyy = ((-1./12.) * P_present[zxy(z, x, y + 2)] + (4./3.) * P_present[zxy(z, x, y + 1)] - (5./2.) * P_present[zxy(z, x, y)] + (4./3.) * P_present[zxy(z, x, y - 1)] - (1./12.) * P_present[zxy(z, x, y - 2)]) / (infoF32.dy * infoF32.dy);
    }


    lap[zxy(z, x, y)] = pxx + pyy + pzz;
}

@compute
@workgroup_size(wsz, wsx, wsy)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let y: i32 = i32(index.z);

    P_future[zxy(z, x, y)] = (c * c) * lap[zxy(z, x, y)] * (infoF32.dt * infoF32.dt);

    P_future[zxy(z, x, y)] += ((2. * P_present[zxy(z, x, y)]) - P_past[zxy(z, x, y)]);

    if (x == infoI32.source_x && y == infoI32.source_y && z == infoI32.source_z)
    {
        P_future[zxy(z, x, y)] += source[infoI32.i];
    }

    P_past[zxy(z, x, y)] = P_present[zxy(z, x, y)];
    P_present[zxy(z, x, y)] = P_future[zxy(z, x, y)];
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
