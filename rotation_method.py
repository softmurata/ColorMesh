import torch
import math

def rotation_around_grid_centroid(view_params):
    # transformation_matrix => (4, 4)
    batch_size = view_params.shape[0]
    azimuth = view_params[:, 0]
    elevation = view_params[:, 1]
    scale = view_params[:, 2]
    azimuth = (azimuth - math.pi * 0.5)  # Ofset azimuth by -90 degree to match with VTk coordinate system
    # reshape for transformation matrix format
    azimuth = azimuth.reshape(batch_size, 1, 1)
    elevation = elevation.reshape(batch_size, 1, 1)
    scale = scale.reshape(batch_size, 1, 1)

    ones = torch.ones_like(azimuth)
    zeros = torch.zeros_like(azimuth)

    # Y axis rotation(azimuth angle)
    rotY = torch.cat([torch.cat([torch.cos(azimuth), zeros, -torch.sin(azimuth), zeros], dim=2),
                      torch.cat([zeros, ones, zeros, zeros], dim=2),
                      torch.cat([torch.sin(azimuth), zeros, torch.cos(azimuth), zeros], dim=2),
                      torch.cat([zeros, zeros, zeros, ones], dim=2)], dim=1)  # (batch_size, 4, 4)

    # Z axis rotation(elevation angle)
    rotZ = torch.cat([torch.cat([torch.cos(elevation), torch.sin(elevation), zeros, zeros], dim=2),
                      torch.cat([-torch.sin(elevation), torch.cos(elevation), zeros, zeros], dim=2),
                      torch.cat([zeros, zeros, ones, zeros], dim=2),
                      torch.cat([zeros, zeros, zeros, ones], dim=2)], dim=1)
    transformation_matrix = torch.matmul(rotY, rotZ)

    scale_matrix = torch.cat([torch.cat([scale, zeros, zeros, zeros], dim=2),
                              torch.cat([zeros, scale, zeros, zeros], dim=2),
                              torch.cat([zeros, zeros, scale, zeros], dim=2),
                              torch.cat([zeros, zeros, zeros, ones], dim=2)], dim=1)

    return transformation_matrix, scale_matrix

def create_voxel_grid(depth, width, height, homogeneous=True):
    z_t, y_t, x_t = torch.meshgrid([torch.arange(0, depth),
                                    torch.arange(0, height),
                                    torch.arange(0, width)])
    # reshape (1, depth * width * height)
    z_t_flat = z_t.reshape(1, -1)
    x_t_flat = x_t.reshape(1, -1)
    y_t_flat = y_t.reshape(1, -1)

    grid = torch.cat([x_t_flat, y_t_flat, z_t_flat], dim=0)

    if homogeneous:
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([grid, ones], dim=0)  # (4, 128 * 128 * 128)

    return grid

def interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat, out_size):
    # print('interpolation part start')
    batch_size, n_channels, depth, height, width = voxel_array.shape
    x = x_s_flat.type(torch.float32)
    y = y_s_flat.type(torch.float32)
    z = z_s_flat.type(torch.float32)

    _, out_channels, out_depth, out_height, out_width = out_size

    max_x = height - 1
    max_y = width - 1
    max_z = depth - 1

    # do sampling
    # axis = 0
    x0 = torch.floor(x).type(torch.int32)
    y0 = torch.floor(y).type(torch.int32)
    z0 = torch.floor(z).type(torch.int32)

    # axis = 1
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    z0 = torch.clamp(z0, 0, max_z)
    z1 = torch.clamp(z1, 0, max_z)

    batch_range = torch.arange(batch_size) * width * height * depth
    n_repeats = out_depth * out_height * out_height
    rep = torch.ones([1, n_repeats], dtype=torch.int32)
    base = torch.matmul(batch_range.reshape(-1, 1).float(), rep.float())
    base = base.reshape(-1)

    # find z element
    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height

    # find the y element based on z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # find index element
    # z = 0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1

    # z = 1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    # voxel flatten
    voxel_flat = voxel_array.permute(0, 2, 3, 4, 1).reshape(-1, n_channels)  # (batch_size * depth * height * width, n_channels)

    Ia = voxel_flat[idx_a.type(torch.int64)]  # (batch_size * height * width * depth, 1)
    Ib = voxel_flat[idx_b.type(torch.int64)]
    Ic = voxel_flat[idx_c.type(torch.int64)]
    Id = voxel_flat[idx_d.type(torch.int64)]
    Ie = voxel_flat[idx_e.type(torch.int64)]
    If = voxel_flat[idx_f.type(torch.int64)]
    Ig = voxel_flat[idx_g.type(torch.int64)]
    Ih = voxel_flat[idx_h.type(torch.int64)]

    # convert float type
    x0_f = x0.type(torch.float32)
    x1_f = x1.type(torch.float32)
    y0_f = y0.type(torch.float32)
    y1_f = y1.type(torch.float32)
    z0_f = z0.type(torch.float32)
    z1_f = z1.type(torch.float32)

    # first slice at z=0
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y) * (z1_f - z)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f) * (z1_f - z)), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y) * (z1_f - z)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f) * (z1_f - z)), 1)

    # first slice at z=1
    we = torch.unsqueeze(((x1_f - x) * (y1_f - y) * (z - z0_f)), 1)
    wf = torch.unsqueeze(((x1_f - x) * (y - y0_f) * (z - z0_f)), 1)
    wg = torch.unsqueeze(((x - x0_f) * (y1_f - y) * (z - z0_f)), 1)
    wh = torch.unsqueeze(((x - x0_f) * (y - y0_f) * (z - z0_f)), 1)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd + Ie * we + If * wf + Ig * wg + Ih * wh
    # print(output.shape, n_channels)

    return output


def resampling(voxel_array, transformation_matrix, scale_matrix, size, new_size):
    """

    :param voxel_array: shape => (batch_size, n_channels, depth, height, width)
    :param transformation_matrix:
    :param scale_matrix:
    :param size:
    :param new_size:
    :return:
    """

    batch_size = voxel_array.shape[0]
    n_channels = voxel_array.shape[1]
    # initialize new voxel grid
    new_voxel_grid = torch.zeros(batch_size, new_size, new_size, new_size)

    T = torch.Tensor([[1, 0, 0, -size * 0.5],
                      [0, 1, 0, -size * 0.5],
                      [0, 0, 1, -size * 0.5],
                      [0, 0, 0, 1]])
    T = T.reshape(1, 4, 4)
    T = T.repeat(batch_size, 1, 1)

    T_new_inv = torch.Tensor([[1, 0, 0, new_size * 0.5],
                              [0, 1, 0, new_size * 0.5],
                              [0, 0, 1, new_size * 0.5],
                              [0, 0, 0, 1]])
    T_new_inv = T_new_inv.reshape(1, 4, 4).repeat(batch_size, 1, 1)

    total_M = torch.matmul(torch.matmul(torch.matmul(T_new_inv.float(), scale_matrix.float()),
                                        transformation_matrix.float()), T)
    total_M = torch.inverse(total_M)
    # delete homogeneous dimension
    total_M = total_M[:, 0:3, :]

    # create voxel grid
    grid = create_voxel_grid(new_size, new_size, new_size, homogeneous=True)

    grid_transform = torch.matmul(total_M.float(), grid.float())  # (batch_size, 3, 128 * 128 * 128)

    x_s_flat = grid_transform[:, 0, :].reshape(-1)  # (batch_size * 128 * 128 * 128)
    y_s_flat = grid_transform[:, 1, :].reshape(-1)  # (batch_size * 128 * 128 * 128)
    z_s_flat = grid_transform[:, 2, :].reshape(-1)  # (batch_size * 128 * 128 * 128)

    out_size = [batch_size, n_channels, new_size, new_size, new_size]

    input_transformed = interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat, out_size)  # (batch_size * depth * height * width, 1)
    new_voxel_grid = input_transformed.reshape(batch_size, n_channels, new_size, new_size, new_size)

    return new_voxel_grid

def rotation_resampling(voxel_array, view_params, size=64, new_size=128):
    """

    :param voxel_array: voxel grid => (b, c, d, h, w)
    :param view_params: (azimuth, elevation, scale) => (b, 3)
    :param size: base voxel grid size
    :param new_size: new voxel grid size
    :return:
    """

    # compute transformation matrix from (azimuth, elevation, scale)
    transformation_matrix, scale_matrix = rotation_around_grid_centroid(view_params)
    # new size voxel grid
    new_voxel_grid = resampling(voxel_array, transformation_matrix, scale_matrix, size=size, new_size=new_size)

    return new_voxel_grid


def match_voxel_to_image(rotation_model):
    reverse = torch.arange(rotation_model.shape[-1] - 1, -1, -1)
    rotation_model = rotation_model[:, :, :, :, reverse]

    return rotation_model


if __name__ == '__main__':
    import numpy as np
    size = 64
    new_size = 128
    batch_size = 2

    azimuth_low = 0
    azimuth_high = 359
    elevation_low = 10
    elevation_high = 170
    scale_low = 3
    scale_high = 6.3

    voxel_array = torch.randn(batch_size, 1, size, size, size)
    view_params = np.array([[np.random.randint(azimuth_low, azimuth_high) * np.pi / 180.0,
                             np.random.randint(elevation_low, elevation_high) * np.pi / 180.0,
                             1] for _ in range(batch_size)])
    view_params = torch.from_numpy(view_params)

    new_voxel_grid = rotation_resampling(voxel_array, view_params, size, new_size)
    print('new voxel grid shape:', new_voxel_grid.shape)
