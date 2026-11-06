"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

quicklook amr_grid script
Patch outlines per AMR level in a slab around a centre (graphics.plot_AMR_grid) and the fraction of the volume of the
window in leaf cells of each level. Run from the repository root:

    python quicklook/amr_grid.py --sim-dir SIM --params masclet_parameters.json --it 1350 --level 9
                                 --center peak --width 8 [--slab 0.5] [--axis z] [--out FOLDER]

The window is the cube of side --width (2R with --sphere R) around the centre; the slab has thickness --slab along
--axis. Leaf cells are counted by their centre, so the leaf volume differs from width^3 by partial cells at the edges.

Created by Marco Molina Pradillo
"""

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# QUICKLOOK COMMON MODULE (it puts the masclet_framework package in sys.path)
import common
from masclet_framework import graphics


# FUNCTIONS DEFINED IN THIS MODULE

PLANES = {'z': ('xy', (0, 1)), 'y': ('xz', (0, 2)), 'x': ('yz', (1, 2))}
LEVEL_CMAP = 'inferno'


def level_colors(max_level, min_level=0):
    """
    Colours of the levels, as plot_AMR_grid samples them from LEVEL_CMAP.

    Author: Marco Molina Pradillo
    """
    return plt.get_cmap(LEVEL_CMAP)(np.linspace(0.2, 0.8, 1 + max_level - min_level))


def plot_grid(args, grid, centre, width, fractions, leaf_volume, centre_info, width_label):
    """
    Figure: patch outlines in the slab (left) and leaf volume fraction per level in the window (right).

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        centre: centre (cMpc)
        width: side of the window (cMpc)
        fractions: leaf volume fraction of each level in the window
        leaf_volume: leaf volume found in the window (cMpc^3)
        centre_info: description of the centre (str)
        width_label: side of the window asked for, for the file name (cMpc)

    Returns:
        path of the figure

    Author: Marco Molina Pradillo
    """
    plane, axes = PLANES[args.axis]
    los = 'xyz'.index(args.axis)
    half = np.full(3, width / 2)
    half[los] = args.slab / 2
    box = [v for c, h in zip(centre, half) for v in (c - h, c + h)]
    colors = level_colors(args.level)
    with common.figure_style():
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})
        graphics.plot_AMR_grid(args.it, path=args.sim_dir, parameters_path=args.params_dir, L=grid['size'],
                               nmax=grid['nmax'], ax=ax0, box=box, proyection=plane, cm=LEVEL_CMAP,
                               linewidth_max=args.linewidth_max, linewidth_min=args.linewidth_min, min_level=0,
                               max_level=args.level)
        ax0.set_aspect('equal')
        ax0.set_xlabel('%s [cMpc]' % 'xyz'[axes[0]])
        ax0.set_ylabel('%s [cMpc]' % 'xyz'[axes[1]])
        common.add_scale_bar(ax0, width, color='black')
        ax0.set_title('patches intersecting the slab |%s - %.3f| < %.3g cMpc' % (args.axis, centre[los], args.slab / 2))
        ax0.legend(handles=[Line2D([], [], color=colors[l], linewidth=1.5, label='L%d' % l)
                            for l in range(args.level + 1)],
                   loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, handlelength=1.2)

        levels = np.arange(len(fractions))
        shown = fractions > 0
        ax1.bar(levels[shown], fractions[shown], color=colors[levels[shown]], width=0.7)
        for l in levels[shown]:
            ax1.annotate('%.2g' % fractions[l], (l, fractions[l]), textcoords='offset points', xytext=(0, 3),
                         ha='center', fontsize=8)
        ax1.set_yscale('log')
        ax1.set_ylim(min(fractions[shown].min() / 3, 0.1), 2)
        ax1.set_xticks(levels)
        ax1.set_xticklabels(['L%d' % l for l in levels])
        ax1.set_xlabel('AMR level')
        ax1.set_ylabel('leaf volume fraction')
        ax1.set_title('window %.3g$^3$ cMpc$^3$ (leaf volume %.4g cMpc$^3$)' % (width, leaf_volume))
        ax1.grid(True, axis='y', which='major', color='0.85', linewidth=0.6)
        fig.suptitle(common.figure_title(args, grid)
                     + '\ncentre (%.3f, %.3f, %.3f) cMpc: %s' % (tuple(centre) + (centre_info,)), fontsize=10)
        fig.tight_layout()
        path = common.output_name(args, '%s_amr_%s_w%g_slab%g' % ('box' if args.whole_box else 'win', args.axis,
                                                                  width_label, args.slab))
        fig.savefig(path)
        plt.close(fig)
    return path


def main():
    parser = common.build_parser('AMR patches per level in a slab and leaf volume fraction per level in a window.')
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='z', help='normal of the slab (default z)')
    parser.add_argument('--slab', type=float, default=0.5, help='thickness of the slab in cMpc (default 0.5)')
    parser.add_argument('--linewidth-max', type=float, default=1.5,
                        help='line width of the level-0 outlines (plot_AMR_grid divides it by (level + 1)^2)')
    parser.add_argument('--linewidth-min', type=float, default=0.25, help='minimum line width of the outlines')
    args = common.parse_arguments(parser)
    timer = common.Timer()

    grid = common.read_grid(args)
    gas = common.read_gas(args, pressure=False, temperature=False, magnetic=False)
    timer.step('read grids + clus (delta, cr0amr, solapst) up to L%d' % args.level)
    centre, centre_info = common.find_centre(args, grid, common.derived_fields(gas)['rho'], gas)
    timer.step('centre (%.4f, %.4f, %.4f): %s' % (tuple(centre) + (centre_info,)))

    width_label, _ = common.view_window(args, grid, centre)
    width = width_label
    box = [v for c in centre for v in (c - width / 2, c + width / 2)]
    if not args.whole_box and (np.abs(centre) + width / 2 > grid['size'] / 2).any():
        raise SystemExit('The window leaves the simulation box; reduce --width or move the centre')
    fractions, leaf_volume = common.level_volume_fractions(grid, gas, box, args.level)
    timer.step('leaf volume fractions: ' + ' '.join('L%d %.4g' % (l, f) for l, f in enumerate(fractions)))
    path = plot_grid(args, grid, centre, width, fractions, leaf_volume, centre_info, width_label)
    timer.step('figure %s' % path)


if __name__ == '__main__':
    main()
