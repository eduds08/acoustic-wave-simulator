import wgpu

if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

import matplotlib.pyplot as plt
import numpy as np

""" Parameters """

grid_size_z = 300
grid_size_x = 300

dz = 1
dx = 1

dt = 0.001

c = np.zeros((grid_size_z, grid_size_x), dtype=np.float32)
c += 450.0

total_time = 500
time = np.linspace(0, total_time * dt, total_time, dtype=np.float32)

source_z = np.int32(grid_size_z / 2)
source_x = np.int32(grid_size_x / 2)

f0 = 10
t0 = 2 / f0
source = np.array(-8. * (time - t0) * f0 * (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2)), dtype=np.float32)

""" Workgroup Size """

wsz = 1
for n in range(15, 0, -1):
    if (grid_size_z % n) == 0:
        wsz = n
        break

wsx = 1
for n in range(15, 0, -1):
    if (grid_size_x % n) == 0:
        wsx = n
        break


def sim_webgpu():
    p_present = np.zeros((grid_size_z, grid_size_x), dtype=np.float32)
    p_past = np.zeros((grid_size_z, grid_size_x), dtype=np.float32)
    p_future = np.zeros((grid_size_z, grid_size_x), dtype=np.float32)
    lap = np.zeros((grid_size_z, grid_size_x), dtype=np.float32)

    device = wgpu.utils.get_default_device()

    info_int = np.array(
        [
            np.int32(grid_size_z), np.int32(grid_size_x), np.int32(source_z), np.int32(source_x), 0
        ],
        dtype=np.int32
    )

    info_float = np.array(
        [
            np.float32(dz), np.float32(dx), np.float32(dt)
        ],
        dtype=np.float32
    )

    cshader = None
    with open('2d_acoustic_wave_shader.wgsl') as shader_file:
        cshader_string = shader_file.read().replace('wsz', f'{wsz}').replace('wsx', f'{wsx}')
        cshader = device.create_shader_module(code=cshader_string)

    # Info Int
    b0 = device.create_buffer_with_data(data=info_int, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_SRC)
    # Info Float
    b1 = device.create_buffer_with_data(data=info_float, usage=wgpu.BufferUsage.STORAGE |
                                                              wgpu.BufferUsage.COPY_SRC)
    # Source
    b2 = device.create_buffer_with_data(data=source, usage=wgpu.BufferUsage.STORAGE |
                                                           wgpu.BufferUsage.COPY_SRC)
    # Velocity Map
    b3 = device.create_buffer_with_data(data=c, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_SRC)
    # Pressure Present
    b4 = device.create_buffer_with_data(data=p_present, usage=wgpu.BufferUsage.STORAGE |
                                                        wgpu.BufferUsage.COPY_SRC)
    # Pressure Past
    b5 = device.create_buffer_with_data(data=p_past, usage=wgpu.BufferUsage.STORAGE |
                                                      wgpu.BufferUsage.COPY_SRC)

    # Pressure Future
    b6 = device.create_buffer_with_data(data=p_future, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_DST |
                                                            wgpu.BufferUsage.COPY_SRC)

    # Laplacian
    b7 = device.create_buffer_with_data(data=lap, usage=wgpu.BufferUsage.STORAGE |
                                                        wgpu.BufferUsage.COPY_DST |
                                                        wgpu.BufferUsage.COPY_SRC)

    binding_layouts_params = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type":  wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 3,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
    ]
    binding_layouts_sim_arrays = [
        {
            "binding": 4,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 5,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 6,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 7,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]

    bindings_params = [
        {
            "binding": 0,
            "resource": {"buffer": b0, "offset": 0, "size": b0.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b1, "offset": 0, "size": b1.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b2, "offset": 0, "size": b2.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b3, "offset": 0, "size": b3.size},
        },
    ]
    bindings_sim_arrays = [
        {
            "binding": 4,
            "resource": {"buffer": b4, "offset": 0, "size": b4.size},
        },
        {
            "binding": 5,
            "resource": {"buffer": b5, "offset": 0, "size": b5.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b6, "offset": 0, "size": b6.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b7, "offset": 0, "size": b7.size},
        },
    ]

    bind_group_layout_0 = device.create_bind_group_layout(entries=binding_layouts_params)
    bind_group_layout_1 = device.create_bind_group_layout(entries=binding_layouts_sim_arrays)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout_0,
                                                                        bind_group_layout_1])
    bind_group_0 = device.create_bind_group(layout=bind_group_layout_0, entries=bindings_params)
    bind_group_1 = device.create_bind_group(layout=bind_group_layout_1, entries=bindings_sim_arrays)

    compute_sim = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "sim"},
    )

    compute_lap = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "laplacian_5_operator"},
    )

    compute_incr = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "incr_time"},
    )

    for i in range(total_time):
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()

        compute_pass.set_bind_group(0, bind_group_0, [], 0, 999999)  # last 2 elements not used
        compute_pass.set_bind_group(1, bind_group_1, [], 0, 999999)  # last 2 elements not used

        compute_pass.set_pipeline(compute_lap)
        compute_pass.dispatch_workgroups(grid_size_z // wsz, grid_size_x // wsx)

        compute_pass.set_pipeline(compute_sim)
        compute_pass.dispatch_workgroups(grid_size_z // wsz, grid_size_x // wsx)

        compute_pass.set_pipeline(compute_incr)
        compute_pass.dispatch_workgroups(1)

        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

    out = device.queue.read_buffer(b6).cast("f")  # reads from buffer 6 (Pressure Field Future)
    adapter_info = device.adapter.request_adapter_info()

    return np.asarray(out).reshape((grid_size_z, grid_size_x)), adapter_info["device"]


p_fut, gpu_str = sim_webgpu()
print(gpu_str)

fig = plt.figure()
plt.imshow(p_fut)
plt.show()
