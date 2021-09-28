import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product, count

from matplotlib.colors import LinearSegmentedColormap


# it produce more vectors pointing diagonally than vectors pointing along
# an axis
# # generate uniform unit vectors
# def generate_unit_vectors(n):
#     'Generates matrix NxN of unit length vectors'
#     v = np.random.uniform(-1, 1, (n, n, 2))
#     l = np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2).reshape(n, n, 1)
#     v /= l
#     return v

def generate_unit_vectors(n,m):
    'Generates matrix NxN of unit length vectors'
    phi = np.random.uniform(0, 2*np.pi, (n, m))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v


# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t


def generate_2D_perlin_noise(size = (200,200), ns=1):
    '''
    generate_2D_perlin_noise(size, ns)
    Generate 2D array of size x size filled with Perlin noise.
    Parameters
    ----------
    size : int or (int, int)
        Size of 2D array size x size.
    ns : int
        Distance between nodes.
    Returns
    -------
    m : ndarray
        The 2D array filled with Perlin noise.
    '''
    if type(size) == int:
        size = (size, size)
    nc = [int(size[0] / ns), int(size[1] / ns)]  # number of nodes
    grid_size_h = int(size[0] / ns + 1)  # number of points in grid
    grid_size_w = int(size[1] / ns + 1)  # number of points in grid

    # generate grid of vectors
    v = generate_unit_vectors(grid_size_h, grid_size_w)

    # generate some constans in advance
    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)
    bd, br = np.arange(ns), np.arange(-ns, 0, 1)

    # vectors from each of the 4 nearest nodes to a point in the NSxNS patch
    vd = np.zeros((ns, ns, 4, 1, 2))
    # for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
    vd[:, :, 0, 0] = np.stack(np.meshgrid(bd, ad, indexing='xy'), axis=2)
    vd[:, :, 1, 0] = np.stack(np.meshgrid(br, ad, indexing='xy'), axis=2)
    vd[:, :, 2, 0] = np.stack(np.meshgrid(bd, ar, indexing='xy'), axis=2)
    vd[:, :, 3, 0] = np.stack(np.meshgrid(br, ar, indexing='xy'), axis=2)

    # interpolation coefficients
    d = qz(np.stack((np.zeros((ns, ns, 2)),
                     np.stack(np.meshgrid(ad, bd, indexing='ij'), axis=2)),
           axis=2)/ns)
    dd = np.stack(np.meshgrid(ad, bd, indexing='ij'), axis=2)
    dd = dd.astype('float')
    d[:, :, 0] = 1 - d[:, :, 1]
    # make copy and reshape for convenience
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)
    # print(d0,d1)

    # make an empy matrix
    m = np.zeros((size[0], size[1]))
    # reshape for convenience
    t = m.reshape(nc[0], ns, nc[1], ns)

    # calculate values for a NSxNS patch at a time
    for i in np.arange(nc[0]):
        for j in np.arange(nc[1]):  # loop through the grid
            # get four node vectors
            av = v[i:i+2, j:j+2].reshape(4, 2, 1)
            # 'vector from node to point' dot 'node vector'
            at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
            # horizontal and vertical interpolation
            t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)

    return m
if __name__ == "__main__":
    img = generate_2D_perlin_noise(200, 20)
    plt.figure()
    plt.imshow(img, cmap=cm.gray)
    img = generate_2D_perlin_noise((200,300), 10)
    print(type(img), img.shape, img.min(), img.max())
    plt.figure()
    plt.imshow(img, cmap=cm.gray)
    plt.axis('off')
    img = generate_2D_perlin_noise((200,50), 25)
    print(type(img), img.shape, img.min(), img.max())
    plt.figure()
    plt.imshow(img, cmap=cm.gray)
    plt.axis('off')
    plt.figure()
    plt.imshow(img>3, cmap=cm.gray)
    plt.figure()
    plt.imshow(img>1, cmap=cm.gray)

    # generate "sky"
    #img0 = generate_2D_perlin_noise(400, 80)
    #img1 = generate_2D_perlin_noise(400, 40)
    #img2 = generate_2D_perlin_noise(400, 20)
    #img3 = generate_2D_perlin_noise(400, 10)
    #
    #img = (img0 + img1 + img2 + img3) / 4
    #cmap = LinearSegmentedColormap.from_list('sky',
    #                                         [(0, '#0572D1'),
    #                                          (0.75, '#E5E8EF'),
    #                                          (1, '#FCFCFC')])
    #img = cm.ScalarMappable(cmap=cmap).to_rgba(img)
    #plt.imshow(img)