"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

quicklook radial_profiles script
Radial profiles (profiles.radial_profile, directional sampling) of the gas density, the dark matter density, T, |B|
and beta = P_gas / P_B around a centre, one panel per quantity, and a text table with the same profiles. Run from the
repository root:

    python quicklook/radial_profiles.py --sim-dir SIM --params masclet_parameters.json --it 1350 --level 7
                                        --center peak --sphere 4 [--nbins 30] [--out FOLDER]

Profiles: densities are the mean over directions, T is weighted by the gas density, |B| is the rms (sqrt of the mean
B^2) and beta = 2 <P> / <B^2>. Bins are logarithmic from the cell size at --res-level to the radius of the region.

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
from masclet_framework import profiles


# FUNCTIONS DEFINED IN THIS MODULE

def compute_profiles(args, grid, gas, dm, quantities, centre, rmin, rmax):
    """
    Radial profiles of the given quantities around a centre.

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        gas: dictionary of common.read_gas
        dm: rho_DM / rho_B (common.read_dm), or None
        quantities: keys of common.QUANTITIES
        centre: centre (cMpc)
        rmin, rmax: radial range (cMpc)

    Returns:
        radii of the bins (cMpc) and dictionary of profiles

    Author: Marco Molina Pradillo
    """
    geometry = (grid['npatch'], grid['patchrx'], grid['patchry'], grid['patchrz'], grid['patchnx'], grid['patchny'],
                grid['patchnz'], grid['size'], grid['nmax'])
    der = common.derived_fields(gas)

    def one(field, weight=None):
        return profiles.radial_profile(field, centre[0], centre[1], centre[2], *geometry, weight_field=weight,
                                       rmin=rmin, rmax=rmax, nbins=args.nbins, logbins=True, maxl=args.level,
                                       interpolate=True, average='mean', Ncostheta=args.ncostheta, Nphi=args.nphi)

    out = {}
    if 'gas' in quantities:
        out['gas'], r = one(der['rho'])
    if 'dm' in quantities:
        out['dm'], r = one(dm)
    if 'temp' in quantities:
        out['temp'], r = one(gas['temp'], weight=der['rho'])
    if 'B' in quantities or 'beta' in quantities:
        B2, r = one(der['B2'])
        if 'B' in quantities:
            out['B'] = np.sqrt(B2) * common.MAGNETICFIELD_TO_MICROGAUSS
        if 'beta' in quantities:
            out['beta'] = 2 * one(gas['pres'])[0] / B2
    return np.asarray(r), out


def plot_profiles(args, grid, r, prof, centre, centre_info, radius):
    """
    Figure with one panel per quantity (log-log, radius in ckpc) and text table of the profiles.

    Args:
        args: parsed command line
        grid: dictionary of common.read_grid
        r: radii of the bins (cMpc)
        prof: dictionary of profiles
        centre: centre (cMpc)
        centre_info: description of the centre (str)
        radius: radius of the region (cMpc), for the file name

    Returns:
        path of the figure and of the table

    Author: Marco Molina Pradillo
    """
    keys = [q for q in common.QUANTITIES if q in prof]
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    view = 'profiles_r%g_res%d' % (radius, args.res_level)
    with common.figure_style():
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axs.flat[len(keys):]:
            ax.set_axis_off()
        for ax, q in zip(axs.flat, keys):
            ok = np.isfinite(prof[q]) & (prof[q] > 0)
            ax.plot(1e3 * r[ok], prof[q][ok], color='#2a5d8f', marker='o', markersize=3)
            ax.set_xscale('log')
            ax.set_yscale('log' if common.QUANTITIES[q]['log'] else 'linear')
            ax.set_xlabel('r [ckpc]')
            ax.set_ylabel(common.quantity_label(q))
            ax.grid(True, which='major', color='0.85', linewidth=0.6)
        fig.suptitle(common.figure_title(args, grid, 'radial profiles, %d x %d directions'
                                         % (args.ncostheta, args.nphi))
                     + '\ncentre (%.3f, %.3f, %.3f) cMpc: %s' % (tuple(centre) + (centre_info,)), fontsize=10)
        fig.tight_layout()
        path = common.output_name(args, view)
        fig.savefig(path)
        plt.close(fig)
    table = common.output_name(args, view, ext='txt')
    with open(table, 'w') as f:
        f.write('# %s\n# centre (%.6f, %.6f, %.6f) cMpc: %s\n' % ((common.figure_title(args, grid),) + tuple(centre)
                                                                + (centre_info,)))
        f.write('# r [cMpc]   ' + '   '.join('%s' % q for q in keys)
                + '   (gas, dm: rho/rho_B; temp: K, gas-density weighted; B: rms, microG; beta: 2<P>/<B^2>)\n')
        for i in range(len(r)):
            f.write('%.6e ' % r[i] + ' '.join('%.6e' % prof[q][i] for q in keys) + '\n')
    return path, table


def main():
    parser = common.build_parser('Radial profiles of gas and DM density, T, |B| and beta around a centre.')
    parser.add_argument('--quantities', nargs='+', default=list(common.QUANTITIES), choices=list(common.QUANTITIES))
    parser.add_argument('--nbins', type=int, default=30, help='number of logarithmic radial bins (default 30)')
    parser.add_argument('--ncostheta', type=int, default=20, help='directions in cos(theta) (default 20)')
    parser.add_argument('--nphi', type=int, default=20, help='directions in phi (default 20)')
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
    centre, centre_info = common.find_centre(args, grid, common.derived_fields(gas)['rho'], gas)
    gc.collect()
    timer.step('centre (%.4f, %.4f, %.4f): %s' % (tuple(centre) + (centre_info,)))

    _, radius = common.view_window(args, grid, centre)
    rmin = grid['size'] / grid['nmax'] / 2 ** args.res_level
    r, prof = compute_profiles(args, grid, gas, dm, quantities, centre, rmin, radius)
    timer.step('profiles: %d bins from %.2f to %.0f ckpc' % (len(r), 1e3 * rmin, 1e3 * radius))
    path, table = plot_profiles(args, grid, r, prof, centre, centre_info, radius)
    timer.step('figure %s' % path)


if __name__ == '__main__':
    main()
