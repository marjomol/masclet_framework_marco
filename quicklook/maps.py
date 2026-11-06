"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

quicklook maps script
Slice (graphics.slice_map) or projection (uniform cube of tools.uniform_grid_zoom_interpolate summed along the line of
sight) of the gas density, the dark matter density, T, |B| and beta = P_gas / P_B around a centre, one panel per
quantity. Run from the repository root:

    python quicklook/maps.py --sim-dir SIM --params masclet_parameters.json --it 1350 --level 6 --res-level 6
                             --center peak --width 8 [--mode slice | projection] [--axis z] [--out FOLDER]

Projections: densities are the mean along the line of sight (the sum times dx / depth), T is weighted by the gas mass,
|B| is the rms (sqrt of the mean B^2) and beta = 2 sum(P) / sum(B^2).

Created by Marco Molina Pradillo
"""

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import gc

# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt

# QUICKLOOK COMMON MODULE (it puts the masclet_framework package in sys.path)
import common
from masclet_framework import graphics, tools


# FUNCTIONS DEFINED IN THIS MODULE

# Orientation of the images for each line of sight: normal and north vectors of slice_map, and the horizontal and
# vertical axes of the image (slice_map returns the first index along north x normal and the second along north).
ORIENTATION = {'z': {'normal': [0, 0, 1], 'north': [0, 1, 0], 'axes': (0, 1)},
               'x': {'normal': [1, 0, 0], 'north': [0, 0, 1], 'axes': (1, 2)},
               'y': {'normal': [0, -1, 0], 'north': [0, 0, 1], 'axes': (0, 2)}}
AXIS_NAMES = 'xyz'


def amr_fields(gas, dm, quantities):
    """
    AMR fields (lists of patches) needed by the maps of the given quantities.

    Args:
        gas: dictionary of common.read_gas
        dm: rho_DM / rho_B (common.read_dm), or None
        quantities: keys of common.QUANTITIES to draw

    Returns:
        dictionary with some of rho, rho_dm, temp, rhoT, B2 and pres

    Author: Marco Molina Pradillo
    """
    der = common.derived_fields(gas)
    out = {'rho': der['rho']}
    if 'dm' in quantities:
        out['rho_dm'] = dm
    if 'temp' in quantities:
        out['temp'] = gas['temp']
        out['rhoT'] = [r * t for r, t in zip(der['rho'], gas['temp'])]
    if 'B' in quantities or 'beta' in quantities:
        out['B2'] = der['B2']
    if 'beta' in quantities:
        out['pres'] = gas['pres']
    return out


def slice_images(args, grid, fields, quantities, centre, width, res):
    """
    Slices through the centre, perpendicular to --axis, of the given quantities.

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        fields: dictionary of amr_fields
        quantities: keys of common.QUANTITIES to draw
        centre: centre of the view (cMpc)
        width: side of the window (cMpc), a multiple of 2 res
        res: pixel size (cMpc)

    Returns:
        dictionary of 2d images (first index horizontal, second vertical)

    Author: Marco Molina Pradillo
    """
    o = ORIENTATION[args.axis]
    h, v = np.zeros(3), np.zeros(3)
    h[o['axes'][0]], v[o['axes'][1]] = 1, 1
    # slice_map samples (i - n/2 + 1) res along h and (j - n/2) res along v: shift half a pixel to centre the pixels
    c = np.array(centre) - 0.5 * res * h + 0.5 * res * v
    # and 1e-4 pixels off the cell centres: locate_point (buf=0.5) accepts a point on the centre of the last cell of a
    # patch, and the interpolation then reads one cell past the patch (IndexError in slice_map)
    c = c + 1e-4 * res * (h + v + np.array(o['normal']))
    geometry = (grid['npatch'], grid['patchnx'], grid['patchny'], grid['patchnz'], grid['patchrx'], grid['patchry'],
                grid['patchrz'], grid['size'], grid['nmax'], args.level)

    def one(field):
        return graphics.slice_map(field, o['normal'], o['north'], c[0], c[1], c[2], *geometry, widthN=width,
                                  widthE=width, res=res, interpolate=True, use_tqdm=False)[0]

    images = {}
    if 'gas' in quantities:
        images['gas'] = one(fields['rho'])
    if 'dm' in quantities:
        images['dm'] = one(fields['rho_dm'])
    if 'temp' in quantities:
        images['temp'] = one(fields['temp'])
    if 'B' in quantities or 'beta' in quantities:
        B2 = one(fields['B2'])
        if 'B' in quantities:
            images['B'] = np.sqrt(B2) * common.MAGNETICFIELD_TO_MICROGAUSS
        if 'beta' in quantities:
            images['beta'] = 2 * one(fields['pres']) / B2
    return images


def projection_images(args, grid, fields, quantities, centre, width, depth, whole_box):
    """
    Projections along --axis of the given quantities, from uniform cubes at --res-level (one cube at a time).

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        fields: dictionary of amr_fields
        quantities: keys of common.QUANTITIES to draw
        centre: centre of the view (cMpc)
        width: side of the window (cMpc)
        depth: length of the line of sight (cMpc)
        whole_box: the whole simulation box (no interpolation, which fails at the box edges)

    Returns:
        dictionary of 2d images (first index horizontal, second vertical)

    Author: Marco Molina Pradillo
    """
    size, nmax = grid['size'], grid['nmax']
    los = AXIS_NAMES.index(args.axis)
    half = [width / 2] * 3
    half[los] = depth / 2
    lo, hi = np.array(centre) - half, np.array(centre) + half
    if whole_box:
        limits = (0, nmax - 1) * 3
    else:
        if (lo < -size / 2).any() or (hi > size / 2).any():
            raise SystemExit('The projection window leaves the simulation box; reduce --width / --depth or move the '
                             'centre')
        i1 = np.floor((lo + size / 2) * nmax / size).astype(int)
        i2 = np.ceil((hi + size / 2) * nmax / size).astype(int) - 1
        limits = tuple(int(v) for pair in zip(i1, i2) for v in pair)
    geometry = (grid['npatch'], grid['patchnx'], grid['patchny'], grid['patchnz'], grid['patchrx'], grid['patchry'],
                grid['patchrz'], size, nmax)

    def column(field):
        # sum along the line of sight of the cube cropped to the window, and number of cells summed
        cube, coords = tools.uniform_grid_zoom_interpolate(field, limits, args.res_level, *geometry,
                                                           interpolate=not whole_box, return_coords=True)
        keep = [(np.asarray(c) >= l) & (np.asarray(c) < u) for c, l, u in zip(coords, lo, hi)]
        cube = cube[np.ix_(*keep)]
        out = cube.sum(axis=los, dtype='f8'), cube.shape[los]
        del cube
        gc.collect()
        return out

    images = {}
    rho, n = column(fields['rho'])
    if 'gas' in quantities:
        images['gas'] = rho / n
    if 'dm' in quantities:
        images['dm'] = column(fields['rho_dm'])[0] / n
    if 'temp' in quantities:
        images['temp'] = column(fields['rhoT'])[0] / rho
    if 'B' in quantities or 'beta' in quantities:
        B2 = column(fields['B2'])[0]
        if 'B' in quantities:
            images['B'] = np.sqrt(B2 / n) * common.MAGNETICFIELD_TO_MICROGAUSS
        if 'beta' in quantities:
            images['beta'] = 2 * column(fields['pres'])[0] / B2
    # images with the horizontal axis first (the cube is in x, y, z order)
    axes = [a for a in range(3) if a != los]
    if tuple(axes) != ORIENTATION[args.axis]['axes']:
        images = {k: im.T for k, im in images.items()}
    return images


def plot_maps(args, grid, images, centre, width, centre_info, centre_tag, width_label):
    """
    Figure with one panel per quantity: colour bar with units, scale bar in ckpc and title.

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        images: dictionary of 2d images (first index horizontal)
        centre: centre of the view (cMpc)
        width: side of the window (cMpc)
        centre_info: description of the centre (str)
        centre_tag: label of the centre for the file name (str)
        width_label: side of the window asked for, for the file name (cMpc)

    Returns:
        path of the figure

    Author: Marco Molina Pradillo
    """
    o = ORIENTATION[args.axis]
    ch, cv = centre[o['axes'][0]], centre[o['axes'][1]]
    extent = [ch - width / 2, ch + width / 2, cv - width / 2, cv + width / 2]
    keys = [q for q in common.QUANTITIES if q in images]
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    with common.figure_style():
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.7 * nrow), squeeze=False)
        for ax in axs.flat[len(keys):]:
            ax.set_axis_off()
        for ax, q in zip(axs.flat, keys):
            # zeros (e.g. DM cells without particles) are masked by the log scale: drawn with the lowest colour
            cmap = plt.get_cmap(common.QUANTITIES[q]['cmap']).copy()
            cmap.set_bad(cmap(0.))
            cmap.set_under(cmap(0.))
            im = ax.imshow(images[q].T, extent=extent, cmap=cmap, norm=common.color_norm(q, images[q]))
            common.add_colorbar(fig, ax, im, q)
            common.add_scale_bar(ax, width)
            ax.set_xlabel('%s [cMpc]' % AXIS_NAMES[o['axes'][0]])
            ax.set_ylabel('%s [cMpc]' % AXIS_NAMES[o['axes'][1]])
        what = 'slice' if args.mode == 'slice' else 'projection (depth %.2f cMpc)' % args.depth_used
        fig.suptitle(common.figure_title(args, grid, '%s along %s, res L%d' % (what, args.axis, args.res_level))
                     + '\ncentre (%.3f, %.3f, %.3f) cMpc: %s' % (tuple(centre) + (centre_info,)), fontsize=10)
        fig.tight_layout()
        view = '%s_%s_%s_w%g_res%d' % ('box' if args.whole_box else 'win', args.mode, args.axis, width_label, args.res_level)
        path = common.output_name(args, view, centre_tag)
        fig.savefig(path)
        plt.close(fig)
    return path


def main():
    parser = common.build_parser('Slices or projections of gas and DM density, T, |B| and beta around a centre.')
    parser.add_argument('--mode', choices=('slice', 'projection'), default='slice')
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='z', help='line of sight (default z)')
    parser.add_argument('--depth', type=float, default=None,
                        help='projection depth in cMpc (default: the side of the window)')
    parser.add_argument('--quantities', nargs='+', default=list(common.QUANTITIES), choices=list(common.QUANTITIES))
    args = common.parse_arguments(parser)
    timer = common.Timer()

    quantities = [q for q in args.quantities
                  if not (q in ('B', 'beta') and not args.mascletB) and not (q == 'temp' and not args.cooling)]
    grid = common.read_grid(args)
    gas = common.read_gas(args, pressure='beta' in quantities, temperature='temp' in quantities,
                          magnetic='B' in quantities or 'beta' in quantities)
    timer.step('read grids + clus up to L%d' % args.level)
    dm = common.read_dm(args) if 'dm' in quantities else None
    if dm is not None:
        timer.step('read cldm (deltadm) up to L%d' % args.level)
    fields = amr_fields(gas, dm, quantities)
    centre, centre_info, centre_tag = common.find_centre(args, grid, fields['rho'], gas)
    del gas, dm
    gc.collect()
    timer.step('centre (%.4f, %.4f, %.4f): %s' % (tuple(centre) + (centre_info,)))

    width_label, _ = common.view_window(args, grid, centre)
    res = grid['size'] / grid['nmax'] / 2 ** args.res_level
    width = 2 * res * max(1, round(width_label / (2 * res)))
    if args.mode == 'slice':
        if not args.whole_box and (np.abs(centre) + width / 2 > grid['size'] / 2).any():
            raise SystemExit('The window leaves the simulation box; reduce --width or move the centre')
        images = slice_images(args, grid, fields, quantities, centre, width, res)
        args.depth_used = 0.
    else:
        args.depth_used = width if args.depth is None else args.depth
        images = projection_images(args, grid, fields, quantities, centre, width, args.depth_used, args.whole_box)
    timer.step('%s images %s at res L%d' % (args.mode, 'x'.join(map(str, next(iter(images.values())).shape)),
                                            args.res_level))
    path = plot_maps(args, grid, images, centre, width, centre_info, centre_tag, width_label)
    timer.step('figure %s' % path)


if __name__ == '__main__':
    main()
