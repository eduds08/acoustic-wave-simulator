struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    source_z: i32,
    source_x: i32,
    i: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
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

fn zx(z: i32, x: i32) -> i32 {
    let index = z + x * infoI32.grid_size_z;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn laplacian_5_operator(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    var pzz: f32 = 0.;
    var pxx: f32 = 0.;

    if (z >= 2 && z <= infoI32.grid_size_z - 3)
    {
        pzz = ((-1./12.) * P_present[zx(z + 2, x)] + (4./3.) * P_present[zx(z + 1, x)] - (5./2.) * P_present[zx(z, x)] + (4./3.) * P_present[zx(z - 1, x)] - (1./12.) * P_present[zx(z - 2, x)]) / (infoF32.dz * infoF32.dz);
    }
    if (x >= 2 && x <= infoI32.grid_size_x - 3)
    {
        pxx = ((-1./12.) * P_present[zx(z, x + 2)] + (4./3.) * P_present[zx(z, x + 1)] - (5./2.) * P_present[zx(z, x)] + (4./3.) * P_present[zx(z, x - 1)] - (1./12.) * P_present[zx(z, x - 2)]) / (infoF32.dx * infoF32.dx);
    }

    lap[zx(z, x)] = pzz + pxx;
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    P_future[zx(z, x)] = (c[zx(z, x)] * c[zx(z, x)]) * lap[zx(z, x)] * (infoF32.dt * infoF32.dt);

    P_future[zx(z, x)] += ((2. * P_present[zx(z, x)]) - P_past[zx(z, x)]);

    if (z == infoI32.source_z && x == infoI32.source_x)
    {
        P_future[zx(z, x)] += source[infoI32.i];
    }

    P_past[zx(z, x)] = P_present[zx(z, x)];
    P_present[zx(z, x)] = P_future[zx(z, x)];
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
