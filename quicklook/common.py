"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

quicklook common module
Command line, reading, cleaning, centre finding and figure style shared by the quick-look scripts of this folder
(maps.py, radial_profiles.py, amr_grid.py). Not part of the package: the scripts are run from the repository root
as python quicklook/<script>.py.

Created by Marco Molina Pradillo
"""

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import argparse
import contextlib
import os
import resource
import sys
import time
import warnings

# numpy
import numpy as np
# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# numba
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', NumbaPerformanceWarning)

# MASCLET FRAMEWORK MODULES (the package lives in the parent folder of quicklook/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masclet_framework import parameters, read_masclet, read_asohf, tools


# CONSTANTS

# Magnetic field of the clus files in code units: 1 code unit = 1/13.75 G (as units.magneticfield_to_gauss of MAGNAS)
MAGNETICFIELD_TO_MICROGAUSS = 1e6 / 13.75

# Quantities: label, units of the colour bar, colour map (one sequential map per quantity) and whether it is shown in
# logarithmic scale. beta = P_gas / P_B = 2 P / B^2 with P and B of the clus file in code units (MASCLET writes P and B
# with the same rho_B factor and its total energy carries B~^2 / 2).
QUANTITIES = {'gas': {'label': r'$\rho_{\rm gas}/\rho_B$', 'units': '', 'cmap': 'magma', 'log': True},
              'dm': {'label': r'$\rho_{\rm DM}/\rho_B$', 'units': '', 'cmap': 'bone', 'log': True},
              'temp': {'label': r'$T$', 'units': 'K', 'cmap': 'inferno', 'log': True},
              'B': {'label': r'$|B|$', 'units': r'$\mu$G', 'cmap': 'viridis', 'log': True},
              'beta': {'label': r'$\beta = P_{\rm gas}/P_B$', 'units': '', 'cmap': 'cividis', 'log': True}}


# FUNCTIONS DEFINED IN THIS MODULE

# SECTION: command line

def build_parser(description):
    """
    Command line common to the quick-look scripts. Each script adds its own options to the returned parser.

    Args:
        description: description of the script (str)

    Returns:
        argparse.ArgumentParser with the common options

    Author: Marco Molina Pradillo
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--sim-dir', required=True, help='folder with the grids, clus and cldm files')
    parser.add_argument('--params', required=True, help='masclet_parameters.json, or the folder that contains it')
    parser.add_argument('--it', type=int, required=True, help='iteration')
    parser.add_argument('--level', type=int, required=True, help='maximum AMR level read')
    cooling = parser.add_mutually_exclusive_group()
    cooling.add_argument('--cooling', dest='cooling', action='store_true', default=True,
                         help='record layout of a run with cooling (default)')
    cooling.add_argument('--no-cooling', dest='cooling', action='store_false')
    parser.add_argument('--no-mascletB', dest='mascletB', action='store_false', default=True,
                        help='the clus files have no magnetic field')
    parser.add_argument('--center', nargs='+', default=['peak'], metavar='peak | asohf | X Y Z',
                        help="'peak' (densest leaf cell of the gas), 'asohf' (most massive host halo of the families "
                             "file) or X Y Z in cMpc")
    parser.add_argument('--peak-level', type=int, default=4,
                        help="with --center peak, last level of the gas density searched (its cells keep the mean of "
                             "the finer ones): a coarse level finds the main structure instead of the densest clump "
                             "(default 4)")
    parser.add_argument('--shrink', action='store_true',
                        help="with --center peak, refine the peak with a shrinking sphere on the gas mass")
    parser.add_argument('--shrink-r0', type=float, default=1.0,
                        help='starting radius of the shrinking sphere in cMpc (default 1)')
    parser.add_argument('--halos-dir', default=None, help="folder with the families files (for --center asohf)")
    region = parser.add_mutually_exclusive_group()
    region.add_argument('--width', type=float, default=8., help='side of the window in cMpc (default 8)')
    region.add_argument('--sphere', type=float, default=None, metavar='R', help='radius of the region in cMpc')
    region.add_argument('--whole-box', action='store_true', help='the whole box (the centre is then the box centre)')
    parser.add_argument('--res-level', type=int, default=None,
                        help='AMR level that sets the resolution of the output (default: --level)')
    parser.add_argument('--out', default='.', help='output folder')
    return parser


def parse_arguments(parser):
    """
    Parses the command line and completes the derived settings (paths, simulation name, resolution level).

    Args:
        parser: parser of build_parser with the options of the script

    Returns:
        argparse.Namespace

    Author: Marco Molina Pradillo
    """
    args = parser.parse_args()
    args.sim_dir = os.path.abspath(args.sim_dir)
    params = os.path.abspath(args.params)
    args.params_dir = os.path.dirname(params) if params.endswith('.json') else params
    args.sim_name = os.path.basename(args.sim_dir.rstrip('/'))
    if args.res_level is None:
        args.res_level = args.level
    args.out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(args.out, exist_ok=True)
    return args


# SECTION: reading and cleaning

def read_grid(args):
    """
    Reads the grids file and keeps the patches up to --level, so that every array matches the fields read.

    Args:
        args: parsed command line

    Returns:
        dictionary with irr, t, zeta, nl (levels of the file), npatch (zero above --level), patchnx, patchny, patchnz,
        patchrx, patchry, patchrz, levels (level of each patch), size and nmax

    Author: Marco Molina Pradillo
    """
    irr, t, nl, mapa, zeta, npatch, patchnx, patchny, patchnz, patchrx, patchry, patchrz = \
        read_masclet.read_grids(args.it, path=args.sim_dir, parameters_path=args.params_dir, read_general=True,
                                read_patchnum=True, read_dmpartnum=False, read_patchcellextension=True,
                                read_patchcellposition=False, read_patchposition=True, read_patchparent=False)
    sim_params = parameters.read_parameters_file(path=args.params_dir)
    nmax, size = sim_params['NMAX'], sim_params['SIZE']
    npatch = np.array(npatch).copy()
    npatch[args.level + 1:] = 0
    n = 1 + npatch.sum()
    return {'irr': irr, 't': t, 'zeta': zeta, 'nl': nl, 'npatch': npatch,
            'patchnx': patchnx[:n], 'patchny': patchny[:n], 'patchnz': patchnz[:n],
            'patchrx': patchrx[:n], 'patchry': patchry[:n], 'patchrz': patchrz[:n],
            'levels': tools.create_vector_levels(npatch), 'size': float(size), 'nmax': int(nmax)}


def read_gas(args, pressure=True, temperature=True, magnetic=True):
    """
    Reads the gas (clus) fields used by the quick-look scripts, up to --level. Uncleaned, as read.

    Args:
        args: parsed command line
        pressure, temperature, magnetic: whether P, T and B are read (B only if the run has it, T only with cooling)

    Returns:
        dictionary of fields (lists of patches): delta, cr0amr, solapst and, if read, pres, temp, Bx, By, Bz

    Author: Marco Molina Pradillo
    """
    magnetic = magnetic and args.mascletB
    temperature = temperature and args.cooling
    out = read_masclet.read_clus(args.it, path=args.sim_dir, parameters_path=args.params_dir,
                                 max_refined_level=args.level, output_delta=True, output_v=False,
                                 output_pres=pressure, output_pot=False, output_opot=False, output_temp=temperature,
                                 output_metalicity=False, output_cr0amr=True, output_solapst=True,
                                 is_mascletB=args.mascletB, output_B=magnetic, is_cooling=args.cooling,
                                 verbose=False, use_tqdm=False)
    names = ['delta'] + (['pres'] if pressure else []) + (['temp'] if temperature else []) + ['cr0amr', 'solapst'] \
        + (['Bx', 'By', 'Bz'] if magnetic else [])
    return dict(zip(names, out))


def read_dm(args):
    """
    Reads the dark matter density of the cldm file on the AMR grid, up to --level.

    Args:
        args: parsed command line

    Returns:
        list of patches with rho_DM / rho_B (deltadm + 1)

    Author: Marco Molina Pradillo
    """
    deltadm = read_masclet.read_cldm(args.it, path=args.sim_dir, parameters_path=args.params_dir,
                                     max_refined_level=args.level, output_deltadm=True, output_position=False,
                                     output_velocity=False, output_mass=False, verbose=False, use_tqdm=False)
    if isinstance(deltadm, tuple):
        deltadm = deltadm[0]
    return [d + 1 for d in deltadm]


def derived_fields(gas):
    """
    Per patch fields of the maps and profiles: rho_gas / rho_B, |B|^2 (code units) and, from them, beta = 2 P / B^2.

    Args:
        gas: dictionary of read_gas

    Returns:
        dictionary with rho (1 + delta) and, if B was read, B2 (Bx^2 + By^2 + Bz^2, code units)

    Author: Marco Molina Pradillo
    """
    out = {'rho': [d + 1 for d in gas['delta']]}
    if 'Bx' in gas:
        out['B2'] = [bx ** 2 + by ** 2 + bz ** 2 for bx, by, bz in zip(gas['Bx'], gas['By'], gas['Bz'])]
    return out


def leaf_mask(grid, gas, up_to_level):
    """
    Leaf cells (not refined up to up_to_level and not overlapped) of every patch, as tools.clean_field.

    Args:
        grid: dictionary of read_grid
        gas: dictionary of read_gas (cr0amr, solapst)
        up_to_level: last level (its cells are leaves even if refined further)

    Returns:
        list of boolean arrays, one per patch up to up_to_level

    Author: Marco Molina Pradillo
    """
    ones = [np.ones(c.shape, dtype='f4') for c in gas['cr0amr']]
    clean = tools.clean_field(ones, gas['cr0amr'], gas['solapst'], grid['npatch'], up_to_level=up_to_level)
    return [c > 0 for c in clean]


def cell_centres(grid, ipatch):
    """
    Comoving coordinates of the cell centres of a patch (1d vectors along x, y and z).

    Args:
        grid: dictionary of read_grid
        ipatch: patch index

    Returns:
        x, y, z: 1d numpy arrays

    Author: Marco Molina Pradillo
    """
    size, nmax = grid['size'], grid['nmax']
    if ipatch == 0:
        x = -size / 2 + (np.arange(nmax) + 0.5) * size / nmax
        return x, x, x
    dx = size / nmax / 2 ** grid['levels'][ipatch]
    # patchr*: centre of the first parent (level l-1) cell, so the first cell centre is patchr* - dx / 2
    return tuple(r + (np.arange(n) - 0.5) * dx for r, n in ((grid['patchrx'][ipatch], grid['patchnx'][ipatch]),
                                                            (grid['patchry'][ipatch], grid['patchny'][ipatch]),
                                                            (grid['patchrz'][ipatch], grid['patchnz'][ipatch])))


# SECTION: centre of the view

def densest_leaf_cell(grid, rho, leaves):
    """
    Centre of the densest leaf cell of a field.

    Args:
        grid: dictionary of read_grid
        rho: field (list of patches)
        leaves: leaf cells of each patch (leaf_mask)

    Returns:
        numpy array (x, y, z) in cMpc

    Author: Marco Molina Pradillo
    """
    best, where = -np.inf, None
    for ipatch, (f, m) in enumerate(zip(rho, leaves)):
        if not m.any():
            continue
        fm = np.where(m, f, -np.inf)
        i = np.argmax(fm)
        if fm.flat[i] > best:
            best, where = fm.flat[i], (ipatch, np.unravel_index(i, fm.shape))
    ipatch, (i, j, k) = where
    x, y, z = cell_centres(grid, ipatch)
    return np.array([x[i], y[j], z[k]])


def shrinking_sphere(grid, rho, leaves, centre, r_start, r_min, factor=0.8):
    """
    Shrinking-sphere centre of the mass of a field: the centre of mass of the leaf cells inside a sphere, iterated
    while the radius is reduced by factor down to r_min. Periodic box (minimum image).

    Args:
        grid: dictionary of read_grid
        rho: density field (list of patches)
        leaves: leaf cells of each patch (leaf_mask)
        centre: starting centre (cMpc)
        r_start, r_min: starting and final radius (cMpc)
        factor: radius reduction per iteration

    Returns:
        numpy array (x, y, z) in cMpc

    Author: Marco Molina Pradillo
    """
    size = grid['size']
    centre = np.array(centre, dtype='f8')
    patches = [(ip, cell_centres(grid, ip), (size / grid['nmax'] / 2 ** grid['levels'][ip]) ** 3)
               for ip in range(len(rho)) if leaves[ip].any()]
    r = r_start
    while r >= r_min:
        msum, mpos = 0., np.zeros(3)
        for ip, (x, y, z), dv in patches:
            d = [((c - c0 + size / 2) % size) - size / 2 for c, c0 in zip((x, y, z), centre)]
            if min(np.abs(d[0]).min(), np.abs(d[1]).min(), np.abs(d[2]).min()) > r:
                continue
            dx, dy, dz = np.meshgrid(*d, indexing='ij')
            inside = leaves[ip] & (dx ** 2 + dy ** 2 + dz ** 2 < r ** 2)
            if not inside.any():
                continue
            m = rho[ip][inside] * dv
            msum += m.sum()
            mpos += np.array([(m * dx[inside]).sum(), (m * dy[inside]).sum(), (m * dz[inside]).sum()])
        if msum > 0:
            centre = ((centre + mpos / msum + size / 2) % size) - size / 2
        r *= factor
    return centre


def asohf_centre(halos_dir, it):
    """
    Density peak and radius of the most massive host halo of the ASOHF families file.

    Args:
        halos_dir: folder with the families files
        it: iteration

    Returns:
        numpy array (x, y, z) in cMpc, the halo radius R (cMpc) and its id

    Author: Marco Molina Pradillo
    """
    haloes = read_asohf.read_families(it, path=halos_dir, output_format='dictionaries', exclude_subhaloes=True)
    halo = max(haloes, key=lambda h: h['M'])
    return np.array([halo['x'], halo['y'], halo['z']]), halo['R'], halo['id']


def find_centre(args, grid, rho, gas):
    """
    Centre of the view from --center (and --peak-level, --shrink, --halos-dir).

    Args:
        args: parsed command line
        grid: dictionary of read_grid
        rho: gas density (list of patches)
        gas: dictionary of read_gas (cr0amr, solapst)

    Returns:
        numpy array (x, y, z) in cMpc, a short description of how it was found (str) and the label of the centre
        for the file names (str): cpeakL<peak level> (with 's' when --shrink), casohf, cgiven or cbox

    Author: Marco Molina Pradillo
    """
    if args.whole_box:
        return np.zeros(3), 'box centre', 'cbox'
    kind = args.center[0].lower()
    if kind == 'peak':
        lpeak = min(args.peak_level, args.level)
        c = densest_leaf_cell(grid, rho, leaf_mask(grid, gas, lpeak))
        what = 'densest gas cell up to L%d' % lpeak
        if not args.shrink:
            return c, what, 'cpeakL%d' % lpeak
        dx_res = grid['size'] / grid['nmax'] / 2 ** args.res_level
        c2 = shrinking_sphere(grid, rho, leaf_mask(grid, gas, args.level), c, args.shrink_r0, 4 * dx_res)
        return c2, 'shrinking sphere from the %s (shift %.1f ckpc)' % (what, 1e3 * np.linalg.norm(c2 - c)), \
            'cpeakL%ds' % lpeak
    if kind == 'asohf':
        halos_dir = args.halos_dir or args.sim_dir
        if not os.path.exists(os.path.join(halos_dir, 'families%05d' % args.it)):
            raise SystemExit('--center asohf: no families%05d in %s' % (args.it, halos_dir))
        c, R, hid = asohf_centre(halos_dir, args.it)
        return c, 'ASOHF halo %d (most massive host), R = %.3f cMpc' % (hid, R), 'casohf'
    if len(args.center) == 3:
        return np.array([float(v) for v in args.center]), 'given', 'cgiven'
    raise SystemExit("--center: 'peak', 'asohf' or X Y Z")


def view_window(args, grid, centre):
    """
    Side of the window and radius of the region of the view.

    Args:
        args: parsed command line
        grid: dictionary of read_grid
        centre: centre of the view (cMpc)

    Returns:
        width (side of the window, cMpc) and radius (half of it, or the --sphere radius), in cMpc

    Author: Marco Molina Pradillo
    """
    if args.whole_box:
        return grid['size'], grid['size'] / 2
    if args.sphere is not None:
        return 2 * args.sphere, args.sphere
    return args.width, args.width / 2


# SECTION: leaf volume per level

def level_volume_fractions(grid, gas, box, up_to_level):
    """
    Fraction of the volume of a box in leaf cells of each AMR level (cells counted by their centre).

    Args:
        grid: dictionary of read_grid
        gas: dictionary of read_gas (cr0amr, solapst)
        box: (xmin, xmax, ymin, ymax, zmin, zmax) in cMpc, inside the simulation box
        up_to_level: last level (its cells are leaves even if refined further)

    Returns:
        numpy array of up_to_level + 1 fractions and the leaf volume found (cMpc^3)

    Author: Marco Molina Pradillo
    """
    leaves = leaf_mask(grid, gas, up_to_level)
    vol = np.zeros(up_to_level + 1)
    for ipatch, m in enumerate(leaves):
        if not m.any():
            continue
        x, y, z = cell_centres(grid, ipatch)
        ix, iy, iz = [(c >= lo) & (c < hi) for c, lo, hi in zip((x, y, z), box[0::2], box[1::2])]
        if not (ix.any() and iy.any() and iz.any()):
            continue
        n = m[np.ix_(ix, iy, iz)].sum()
        vol[grid['levels'][ipatch]] += n * (grid['size'] / grid['nmax'] / 2 ** grid['levels'][ipatch]) ** 3
    return vol / vol.sum(), vol.sum()


# SECTION: figure style

@contextlib.contextmanager
def figure_style():
    """
    Local rcParams of the quick-look figures (the global matplotlib settings are not changed).

    Author: Marco Molina Pradillo
    """
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
                         'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
                         'image.origin': 'lower', 'image.interpolation': 'nearest'}):
        yield


def color_norm(q, image, vmin=None, vmax=None):
    """
    Colour normalisation of a quantity: logarithmic or linear (QUANTITIES), limits from the 0.5 and 99.5 percentiles
    of the finite positive values unless given.

    Args:
        q: key of QUANTITIES
        image: 2d array
        vmin, vmax: optional limits

    Returns:
        matplotlib.colors.Normalize instance

    Author: Marco Molina Pradillo
    """
    v = image[np.isfinite(image)]
    if QUANTITIES[q]['log']:
        v = v[v > 0]
    lo, hi = (np.percentile(v, [0.5, 99.5]) if v.size else (1., 10.))
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    return colors.LogNorm(vmin=vmin, vmax=vmax) if QUANTITIES[q]['log'] else colors.Normalize(vmin=vmin, vmax=vmax)


def quantity_label(q):
    """
    Axis or colour-bar label of a quantity, with its units.

    Author: Marco Molina Pradillo
    """
    units = QUANTITIES[q]['units']
    return QUANTITIES[q]['label'] + (' [%s]' % units if units else '')


def add_colorbar(fig, ax, mappable, q):
    """
    Colour bar of a panel, with the label and units of the quantity.

    Author: Marco Molina Pradillo
    """
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(quantity_label(q))
    return cbar


def add_scale_bar(ax, width, color='white'):
    """
    Scale bar in ckpc, about a fifth of the window side (axes in cMpc).

    Args:
        ax: axis
        width: side of the window (cMpc)
        color: colour of the bar and its label

    Author: Marco Molina Pradillo
    """
    target = width * 1e3 / 5
    nice = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    length = nice[nice <= target].max() if (nice <= target).any() else nice[0]
    bar = AnchoredSizeBar(ax.transData, length / 1e3, '%d ckpc' % length, 'lower left', pad=0.5, color=color,
                          frameon=False, size_vertical=width / 200, fontproperties={'size': 9})
    ax.add_artist(bar)


def figure_title(args, grid, extra=''):
    """
    Title of a figure: simulation, iteration, redshift and level.

    Author: Marco Molina Pradillo
    """
    title = '%s | it %d | z = %.3f | L%d' % (args.sim_name, args.it, grid['zeta'], args.level)
    return title + (' | ' + extra if extra else '')


def output_name(args, view, centre_tag, ext='png'):
    """
    Path of a figure or table: <out>/<sim>_itNNNNN_L<level>_<centre_tag>_<view>.<ext>, with the label of the centre
    returned by find_centre.

    Author: Marco Molina Pradillo
    """
    return os.path.join(args.out, '%s_it%05d_L%d_%s_%s.%s' % (args.sim_name, args.it, args.level, centre_tag, view,
                                                              ext))


# SECTION: resources

class Timer:
    """
    Wall time of the steps of a script and peak memory (maxRSS of the process), printed as they finish.

    Author: Marco Molina Pradillo
    """
    def __init__(self):
        self.t0 = self.t = time.time()

    def step(self, what):
        now = time.time()
        print('%-60s %7.1f s   (total %7.1f s, maxRSS %.2f GB)'
              % (what, now - self.t, now - self.t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
              flush=True)
        self.t = now
