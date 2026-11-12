"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

quicklook animate script
Animation over iterations of projections (maps.projection_images) of the gas density and, optionally, |B|. Two views:
    --view box      the whole box, one panel per quantity
    --view window   a window of --width around a centre that follows the cluster
Run from the repository root:

    python quicklook/animate.py --sim-dir SIM --params masclet_parameters.json --it 2550 --level 6 --res-level 5
                                --view window --width 8 --its 300 900 1500 2100 2550 --quantities gas B
                                --halos-dir HALOS [--comoving-B] [--stage all] [--fps 5] [--out FOLDER]

--it is the last iteration of the animation; --its are the frames. With --comoving-B the |B| panel shows a^2 |B|,
a = 1 / (1 + z), in the same units (comoving muG): a frozen-in field diluted by the expansion keeps a^2 |B|, so the
animation shows amplification and not dilution.
Stages, each reading what the previous one wrote in <out>/<name>/, so that a step can be redone alone:
    centres  (window) table centres.txt: the ASOHF host followed backwards from the last iteration with a families
             file and, after it, the gas peak (--peak-level) within --track-radius of the previous centre (so
             that a denser clump elsewhere in the box does not take the centre), refined with a shrinking sphere
             from --shrink-r0; the track, over every output between the first and the last frame, is smoothed with
             a running mean over --smooth outputs, and the steps of the smoothed centre between frames larger than
             --max-step are listed
    images   one images_itNNNNN.npz per frame (float32 maps, z, t and centre; |B| physical)
    render   one PNG per frame with the same colour limits in every frame: --clim, else the lowest 0.5 percentile
             and the highest --pmax percentile of all the frames (or only those of the frame of --clim-it)
    gif      the GIF of the PNGs (Pillow, as matplotlib's PillowWriter)
    mp4      the MP4 of the PNGs (H.264 with the ffmpeg executable, as matplotlib's FFMpegWriter)
    all      the five in order

Created by Marco Molina Pradillo
"""

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import gc
import os
import subprocess

# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
# Pillow (a dependency of matplotlib)
from PIL import Image

# QUICKLOOK MODULES (common puts the masclet_framework package in sys.path)
import common
import maps
from masclet_framework import read_asohf, units


# FUNCTIONS DEFINED IN THIS MODULE

def frame_name(args):
    """
    Name of the animation (folder of its files and name of the GIF): <sim>_<view>_<quantities>_w<width>_L<level>_
    res<res level>, the axis when it is not z.

    Author: Marco Molina Pradillo
    """
    width = 'box' if args.view == 'box' else 'w%g' % args.width
    name = '%s_anim_%s_%s_%s_L%d_res%d' % (args.sim_name, args.view, '-'.join(args.quantities), width, args.level,
                                           args.res_level)
    if args.comoving_B and 'B' in args.quantities:
        name += '_Bcom'
    return name + ('' if args.axis == 'z' else '_' + args.axis)


def frame_image(args, d, q):
    """
    Map of quantity q of a frame as drawn: with --comoving-B, a^2 |B| (a = 1 / (1 + z)) instead of |B|.

    Args:
        args: parsed command line
        d: contents of images_itNNNNN.npz
        q: quantity

    Returns:
        2d array

    Author: Marco Molina Pradillo
    """
    if q == 'B' and args.comoving_B:
        return d[q] / (1 + float(d['zeta'])) ** 2
    return d[q]


def followed_host(halos_dir, its):
    """
    Centre of the ASOHF host followed backwards from the last iteration of its: at each earlier iteration, the most
    massive host within the radius (at least 0.5 cMpc) of the centre of the later one, or the nearest host if none.

    Args:
        halos_dir: folder with the families files
        its: iterations with a families file, in increasing order

    Returns:
        dictionary {it: (centre (cMpc), description (str))}

    Author: Marco Molina Pradillo
    """
    out, prev, rprev = {}, None, None
    for it in sorted(its, reverse=True):
        haloes = read_asohf.read_families(it, path=halos_dir, output_format='dictionaries', exclude_subhaloes=True)
        if not haloes:
            continue
        if prev is None:
            sel = max(haloes, key=lambda h: h['M'])
        else:
            d = np.array([np.linalg.norm(np.array([h['x'], h['y'], h['z']]) - prev) for h in haloes])
            near = [h for h, dd in zip(haloes, d) if dd < max(rprev, 0.5)]
            sel = max(near, key=lambda h: h['M']) if near else haloes[int(np.argmin(d))]
        prev, rprev = np.array([sel['x'], sel['y'], sel['z']]), sel['R']
        out[it] = (prev, 'asohf_host_%d' % sel['id'])
    return out


def near_mask(grid, ipatch, centre, radius):
    """
    Cells of a patch whose centre is within radius of centre (periodic box, minimum image).

    Args:
        grid: dictionary of common.read_grid
        ipatch: patch index
        centre: centre (cMpc)
        radius: radius (cMpc)

    Returns:
        boolean array with the shape of the patch

    Author: Marco Molina Pradillo
    """
    size = grid['size']
    d = [((c - c0 + size / 2) % size) - size / 2 for c, c0 in zip(common.cell_centres(grid, ipatch), centre)]
    dx, dy, dz = np.meshgrid(*d, indexing='ij', sparse=True)
    return dx ** 2 + dy ** 2 + dz ** 2 < radius ** 2


def compute_centres(args):
    """
    Stage centres: table of the raw and smoothed centre of every output between the first and the last frame (window
    view).

    Args:
        args: parsed command line

    Returns:
        path of the table

    Author: Marco Molina Pradillo
    """
    halos_dir = args.halos_dir or args.sim_dir
    # every output between the first and the last frame, not only the frames: the host is followed step by step
    # and the running mean is taken over consecutive outputs
    track = sorted(int(f[5:]) for f in os.listdir(args.sim_dir) if f.startswith('grids') and f[5:].isdigit()
                   and min(args.its) <= int(f[5:]) <= max(args.its))
    with_families = [it for it in track if os.path.exists(os.path.join(halos_dir, 'families%05d' % it))]
    raw = followed_host(halos_dir, with_families) if with_families else {}
    it_last, prev = args.it, None
    for it in track:
        if it in raw:
            prev = raw[it][0]
            continue
        args.it = it
        grid = common.read_grid(args)
        gas = common.read_gas(args, pressure=False, temperature=False, magnetic=False)
        rho = common.derived_fields(gas)['rho']
        lpeak = min(args.peak_level, args.level)
        leaves = common.leaf_mask(grid, gas, lpeak)
        if prev is not None:
            # the densest cell within --track-radius of the previous centre: a denser clump elsewhere in the box
            # must not take the centre away from the cluster
            leaves = [m & near_mask(grid, ip, prev, args.track_radius) if m.any() else m
                      for ip, m in enumerate(leaves)]
        c0 = common.densest_leaf_cell(grid, rho, leaves)
        dx_res = grid['size'] / grid['nmax'] / 2 ** args.res_level
        c = common.shrinking_sphere(grid, rho, common.leaf_mask(grid, gas, args.level), c0, args.shrink_r0, 4 * dx_res)
        raw[it] = (c, 'cpeakL%ds_r%g%s' % (lpeak, args.shrink_r0, '' if prev is None else '_near%g' % args.track_radius))
        prev = c
        del gas, rho, leaves
        gc.collect()
    args.it = it_last
    its = sorted(raw)
    xyz = np.array([raw[it][0] for it in its])
    # running mean over args.smooth outputs (fewer at the ends of the track)
    half = args.smooth // 2
    smooth = np.array([xyz[max(0, i - half):i + half + 1].mean(axis=0) for i in range(len(its))])
    path = os.path.join(args.anim_dir, 'centres.txt')
    with open(path, 'w') as f:
        f.write('# it   x y z (raw, cMpc)   x y z (running mean over %d outputs, cMpc)   step of the raw centre '
                '(ckpc)   source\n' % args.smooth)
        for i, it in enumerate(its):
            step = 0. if i == 0 else 1e3 * np.linalg.norm(xyz[i] - xyz[i - 1])
            f.write('%5d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %7.0f %s\n' % ((it,) + tuple(xyz[i]) + tuple(smooth[i])
                                                                            + (step, raw[it][1])))
    with open(path) as f:
        print(f.read(), flush=True)
    # steps of the centre of the window between consecutive frames (smoothed centre, the one drawn)
    frames = [it for it in args.its if it in raw]
    steps = [(a, b, 1e3 * np.linalg.norm(smooth[its.index(b)] - smooth[its.index(a)]))
             for a, b in zip(frames[:-1], frames[1:])]
    big = [s for s in steps if s[2] > 1e3 * args.max_step]
    print('steps of the smoothed centre between frames: max %.0f ckpc; %d larger than %.0f ckpc%s'
          % (max(s[2] for s in steps) if steps else 0., len(big), 1e3 * args.max_step, ':' if big else ''), flush=True)
    for a, b, s in big:
        print('    it %5d -> %5d  %7.0f ckpc' % (a, b, s), flush=True)
    return path


def read_centres(args):
    """
    Smoothed centres of the frames from the table of compute_centres.

    Author: Marco Molina Pradillo
    """
    table = np.loadtxt(os.path.join(args.anim_dir, 'centres.txt'), usecols=(0, 4, 5, 6))
    return {int(row[0]): row[1:] for row in np.atleast_2d(table)}


def compute_images(args):
    """
    Stage images: projections of the quantities of every frame, saved as images_itNNNNN.npz (float32).

    Args:
        args: parsed command line

    Author: Marco Molina Pradillo
    """
    centres = read_centres(args) if args.view == 'window' else {}
    it_last = args.it
    for it in args.its:
        timer = common.Timer()
        args.it = it
        grid = common.read_grid(args)
        gas = common.read_gas(args, pressure=False, temperature=False, magnetic='B' in args.quantities)
        fields = maps.amr_fields(gas, None, args.quantities)
        del gas
        gc.collect()
        if args.view == 'box':
            centre, width = np.zeros(3), grid['size']
        else:
            centre = centres[it]
            res = grid['size'] / grid['nmax'] / 2 ** args.res_level
            width = 2 * res * max(1, round(args.width / (2 * res)))
        images = maps.projection_images(args, grid, fields, args.quantities, centre, width, width,
                                        args.view == 'box')
        del fields
        gc.collect()
        np.savez(os.path.join(args.anim_dir, 'images_it%05d.npz' % it), centre=centre, width=width, zeta=grid['zeta'],
                 t=grid['t'] * units.time_to_yr / 1e9, **{q: im.astype('f4') for q, im in images.items()})
        timer.step('it %d: images %s at res L%d' % (it, 'x'.join(map(str, next(iter(images.values())).shape)),
                                                     args.res_level))
    args.it = it_last


def colour_limits(args):
    """
    Colour limits of every quantity, the same in all frames: --clim, else the lowest 0.5 percentile and the highest
    --pmax percentile of the frames (all of them, or only that of --clim-it). The percentiles of every frame and the
    limits are printed.

    Returns:
        dictionary {q: (vmin, vmax)}

    Author: Marco Molina Pradillo
    """
    its = args.its if args.clim_it is None else [args.clim_it]
    free = [q for q in args.quantities if q not in args.clim]
    perc = {q: [] for q in free}
    for it in its:
        d = np.load(os.path.join(args.anim_dir, 'images_it%05d.npz' % it))
        for q in free:
            norm = common.color_norm(q, frame_image(args, d, q), pmax=args.pmax)
            perc[q].append((norm.vmin, norm.vmax))
            print('percentiles it %5d %-5s %.6e %.6e' % (it, q, norm.vmin, norm.vmax), flush=True)
    out = {q: (min(p[0] for p in perc[q]), max(p[1] for p in perc[q])) for q in free}
    out.update({q: args.clim[q] for q in args.quantities if q in args.clim})
    for q in args.quantities:
        print('colour limits %-5s %.6e %.6e' % ((q,) + out[q]), flush=True)
    return out


def render_frames(args):
    """
    Stage render: one PNG per frame, same figure size and colour limits in all of them (the GIF needs equal frames).

    Returns:
        list of paths of the PNG frames

    Author: Marco Molina Pradillo
    """
    lims = colour_limits(args)
    o = maps.ORIENTATION[args.axis]
    paths = []
    for it in args.its:
        timer = common.Timer()
        d = np.load(os.path.join(args.anim_dir, 'images_it%05d.npz' % it))
        centre, width = d['centre'], float(d['width'])
        ch, cv = centre[o['axes'][0]], centre[o['axes'][1]]
        extent = [ch - width / 2, ch + width / 2, cv - width / 2, cv + width / 2]
        n = len(args.quantities)
        # fixed layout ('standard' bounding box, no tight_layout): every frame has the same size in pixels
        with common.figure_style(), plt.rc_context({'savefig.bbox': 'standard', 'savefig.dpi': args.dpi}):
            # one panel: a wider figure, for the two title lines
            fig, axs = plt.subplots(1, n, figsize=(4.6 * n if n > 1 else 6.0, 4.4), squeeze=False)
            fig.subplots_adjust(left=0.08 if n > 1 else 0.17, right=0.92 if n > 1 else 0.80, bottom=0.11,
                                top=0.86, wspace=0.35)
            for ax, q in zip(axs.flat, args.quantities):
                cmap = plt.get_cmap(common.QUANTITIES[q]['cmap']).copy()
                cmap.set_bad(cmap(0.))
                cmap.set_under(cmap(0.))
                image = frame_image(args, d, q)
                norm = common.color_norm(q, image, vmin=lims[q][0], vmax=lims[q][1])
                im = ax.imshow(image.T, extent=extent, cmap=cmap, norm=norm)
                cbar = common.add_colorbar(fig, ax, im, q)
                if q == 'B' and args.comoving_B:
                    cbar.set_label(r'$a^2|B|$ (comoving) [$\mu$G]')
                common.add_scale_bar(ax, width)
                ax.set_xlabel('%s [cMpc]' % maps.AXIS_NAMES[o['axes'][0]])
                ax.set_ylabel('%s [cMpc]' % maps.AXIS_NAMES[o['axes'][1]])
            what = 'whole box' if args.view == 'box' else 'centre (%.2f, %.2f, %.2f) cMpc' % tuple(centre)
            fig.suptitle('%s | it %d | z = %.3f | t = %.2f Gyr\nprojection along %s (depth %.3g cMpc), L%d, res L%d, '
                         '%s' % (args.sim_name, it, float(d['zeta']), float(d['t']), args.axis, width, args.level,
                                 args.res_level, what), fontsize=10)
            path = os.path.join(args.anim_dir, 'frame_it%05d.png' % it)
            fig.savefig(path)
            plt.close(fig)
        paths.append(path)
        timer.step('it %d: frame %s' % (it, path))
    return paths


def make_gif(args):
    """
    Stage gif: GIF of the PNG frames of --its (adaptive palette, --fps frames per second, endless loop).

    Returns:
        path of the GIF

    Author: Marco Molina Pradillo
    """
    frames = [Image.open(os.path.join(args.anim_dir, 'frame_it%05d.png' % it)).convert('RGB') for it in args.its]
    frames = [f.convert('P', palette=Image.ADAPTIVE, colors=256) for f in frames]
    path = os.path.join(args.out, frame_name(args) + '.gif')
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000 / args.fps)), loop=0,
                   optimize=False)
    print('gif %s: %d frames, %.1f MB' % (path, len(frames), os.path.getsize(path) / 1e6), flush=True)
    return path


def make_mp4(args):
    """
    Stage mp4: MP4 (H.264, yuv420p, --fps frames per second) of the PNG frames of --its, written by the ffmpeg
    executable (the one of matplotlib's FFMpegWriter) from the PNGs piped in order.

    Returns:
        path of the MP4

    Author: Marco Molina Pradillo
    """
    path = os.path.join(args.out, frame_name(args) + '.mp4')
    # H.264 with yuv420p needs even width and height: pad by one pixel if odd
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '%g' % args.fps, '-c:v', 'png',
           '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
           path]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for it in args.its:
            with open(os.path.join(args.anim_dir, 'frame_it%05d.png' % it), 'rb') as f:
                proc.stdin.write(f.read())
        proc.stdin.close()
    if proc.returncode:
        raise RuntimeError('ffmpeg ended with code %d' % proc.returncode)
    print('mp4 %s: %d frames, %.1f MB' % (path, len(args.its), os.path.getsize(path) / 1e6), flush=True)
    return path


def main():
    parser = common.build_parser('Animation over iterations of projections of the gas density and |B|.')
    parser.add_argument('--view', choices=('box', 'window'), default='window')
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='z', help='line of sight (default z)')
    parser.add_argument('--its', type=int, nargs='+', required=True, help='iterations of the frames')
    parser.add_argument('--quantities', nargs='+', default=['gas', 'B'], choices=['gas', 'B'])
    parser.add_argument('--stage', choices=('centres', 'images', 'render', 'gif', 'mp4', 'all'), default='all')
    parser.add_argument('--track-radius', type=float, default=1.,
                        help='window view, outputs after the last families file: the gas peak is searched only '
                             'within this radius (cMpc) of the centre of the previous output, so that the window '
                             'follows the cluster and does not jump to a denser clump elsewhere (default 1)')
    parser.add_argument('--smooth', type=int, default=5, help='outputs of the running mean of the centres (default 5)')
    parser.add_argument('--max-step', type=float, default=0.3,
                        help='steps of the window centre between frames larger than this (cMpc) are listed '
                             '(default 0.3)')
    parser.add_argument('--comoving-B', dest='comoving_B', action='store_true',
                        help='draw a^2 |B| (comoving), a = 1 / (1 + z), instead of the physical |B|')
    parser.add_argument('--clim', nargs=3, action='append', default=None, metavar=('Q', 'VMIN', 'VMAX'),
                        help='colour limits of quantity Q (default: lowest and highest percentiles of the frames)')
    parser.add_argument('--clim-it', type=int, default=None,
                        help='take the percentile limits from this frame only (default: from all the frames)')
    parser.add_argument('--fps', type=float, default=5., help='frames per second of the GIF and MP4 (default 5)')
    parser.add_argument('--dpi', type=int, default=100, help='dots per inch of the frames (default 100)')
    args = common.parse_arguments(parser)
    args.its = sorted(set(args.its))
    args.clim = {q: (float(lo), float(hi)) for q, lo, hi in (args.clim or [])}
    if args.view == 'box':
        args.whole_box = True
    args.anim_dir = os.path.join(args.out, frame_name(args))
    os.makedirs(args.anim_dir, exist_ok=True)
    timer = common.Timer()
    if args.stage in ('centres', 'all') and args.view == 'window':
        compute_centres(args)
        timer.step('centres of %d frames' % len(args.its))
    if args.stage in ('images', 'all'):
        compute_images(args)
        timer.step('images of %d frames' % len(args.its))
    if args.stage in ('render', 'all'):
        render_frames(args)
        timer.step('PNG of %d frames' % len(args.its))
    if args.stage in ('gif', 'all'):
        make_gif(args)
        timer.step('gif')
    if args.stage in ('mp4', 'all'):
        make_mp4(args)
        timer.step('mp4')


if __name__ == '__main__':
    main()
