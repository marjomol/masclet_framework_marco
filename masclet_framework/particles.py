"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

particles module
Contains several useful functions in order to deal with particles

Created by David Vallés
"""
#  Last update on 25/4/20 0:09

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

# from multiprocessing import Pool

from masclet_framework import cosmo_tools, units, tools

import collections
from scipy import optimize


# FUNCTIONS DEFINED IN THIS MODULE


# SECTION: manipulate MASCLET data
def correct_positive_oripa(oripa, mass):
    """
    Corrects a bug in the simulation's ouput, which caused all oripa to be positive.
    The expected behaviour was that particles which were originally at l=0 retain negative oripa.
    Args:
        oripa: np array contaning the directly read (incorrect) oripa for each particle
        mass: np array containing the mass of each particle

    Returns:
        corrected oripa array, where particles which originally were at l=0 have negative oripa.

    """
    dups = collections.defaultdict(list)
    for i, e in enumerate(oripa):
        dups[e].append(i)

    for thisoripa, v in dups.items():
        if len(v) >= 2:
            if mass[v[0]] > mass[v[1]]:
                oripa[v[0]] = - thisoripa
            else:
                oripa[v[1]] = - thisoripa

    return oripa


# SECTION: compare particles by IDs
def shared_particles(x1, y1, z1, oripa1, rx1, ry1, rz1, r1, x2, y2, z2, oripa2, rx2, ry2, rz2, r2):
    """
    Finds the shared particles between two haloes (of DM particles; although it can be used for any particle simulation)

    Args: for i=1,2
        xi: x position of all the particles at iti
        yi: y position of all the particles at iti
        zi: z position of all the particles at iti
        oripai: unique identifier of the particles at iti
        rxi: x position of the center of the halo i
        ryi: y position of the center of the halo i
        rzi: z position of the center of the halo i
        ri: radius of the halo i where the particles are being considered

    Returns:
        A numpy array containing the intersecting oripas
    """
    inside_1 = (x1 - rx1) ** 2 + (y1 - ry1) ** 2 + (z1 - rz1) ** 2 < r1 ** 2
    inside_1 = oripa1[inside_1]

    inside_2 = (x2 - rx2) ** 2 + (y2 - ry2) ** 2 + (z2 - rz2) ** 2 < r2 ** 2
    inside_2 = oripa2[inside_2]

    intersect = np.intersect1d(inside_1, inside_2, return_indices=False, assume_unique=True)

    return intersect


def shared_mass(x1, y1, z1, m1, oripa1, rx1, ry1, rz1, r1, x2, y2, z2, m2, oripa2, rx2, ry2, rz2, r2):
    """
    Computes the shared mass between two haloes (of DM particles; although it can be used for any particle simulation).
    Units: solar masses.

    Args: for i=1,2
        xi: x position of all the particles at iti
        yi: y position of all the particles at iti
        zi: z position of all the particles at iti
        mi: mass of all the particles at iti
        oripai: unique identifier of the particles at iti
        rxi: x position of the center of the halo i
        ryi: y position of the center of the halo i
        rzi: z position of the center of the halo i
        ri: radius of the halo i where the particles are being considered

    Returns:
        The mass in the older halo (smaller iti) also present in the newer one.
    """
    # checks (just in case)
    if np.isnan(r1):
        r1 = 0
    if np.isnan(r2):
        r2 = 0

    inside_1 = (x1 - rx1) ** 2 + (y1 - ry1) ** 2 + (z1 - rz1) ** 2 < r1 ** 2
    inside_1 = oripa1[inside_1]
    m1 = m1[inside_1]

    inside_2 = (x2 - rx2) ** 2 + (y2 - ry2) ** 2 + (z2 - rz2) ** 2 < r2 ** 2
    inside_2 = oripa2[inside_2]

    _, indices1, _ = np.intersect1d(inside_1, inside_2, return_indices=True, assume_unique=True)

    return m1[indices1].sum() * units.mass_to_sun


# SECTION: find radius from R_\Delta radii definitions
def find_rDelta_eqn_particles(r, Delta, background_density, zeta, clusrx, clusry, clusrz, x, y, z, m, verbose):
    if verbose:
        print('Evaluating at r={:.3f}'.format(r))

    inside_1 = (x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 < r ** 2
    m = m[inside_1].sum() * units.mass_to_sun

    return m - (4 * np.pi / 3) * (r / (1 + zeta)) ** 3 * background_density * Delta


def find_rDelta_particles(Delta, zeta, clusrx, clusry, clusrz, x, y, z, m, h, omega_m, rmin=0.1, rmax=6, rtol=1e-3,
                          verbose=False):
    """
    Finds the value (in Mpc) of r_\Delta, the radius enclosing a mean overdensity (of the DM field, by default) equal
    to Delta times the background density of the universe. By default, it uses the Brent method.

    Args:
        Delta: value of the desired overdensity
        zeta: current redshift
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        x, y, z: vector with the particles' positions
        m: vector with the particles' masses
        h: dimensionless Hubble constant
        omega_m: matter density parameter
        rmin, rmax: starting two points (M(r) - Delta rho_B 4pi/3 r^3) must change sign
        rtol: tolerance in the determination of r
        verbose: if True, prints the patch being opened at a time

    Returns:
        The value of r_\Delta
    """
    background_density = cosmo_tools.background_density(h, omega_m, zeta)
    args = (Delta, background_density, zeta, clusrx, clusry, clusrz, x, y, z, m, verbose)
    try:
        rDelta = optimize.brentq(find_rDelta_eqn_particles, rmin, rmax, args=args, xtol=rtol)
    except ValueError:
        if verbose:
            print('Something might be wrong with this one... Might need further checking.')
        return float('nan')
    if verbose:
        print('Converged!')
    return rDelta


# SECTION: kinematical quantities: angular momenta, shape tensor, etc.
def angular_momentum_particles(x, y, z, vx, vy, vz, m, inside):
    """
    Computes the angular momentum of a particle distribution

    Args:
        x, y, z: (recentered) position of the particles
        vx, vy, vz: components of the particles velocities
        m: particles' masses
        inside: vector of bool values (whether the particles are inside or outside)

    Returns:
        The angular momentum of the given particle distribution. The three components are returned
        in a tuple.

    """
    Lx = (m * (y * vz - z * vy) * inside).sum()
    Ly = (m * (z * vx - x * vz) * inside).sum()
    Lz = (m * (x * vy - y * vx) * inside).sum()

    return Lx, Ly, Lz


def shape_tensor_particles(x, y, z, m, inside):
    """
    Computes the shape tensor of a particle distribution

    Args:
        x, y, z: (recentered) position of the particles
        m: particles' masses
        inside: vector of bool values (whether the particles are inside or outside)

    Returns:
        Shape tensor as a 3x3 matrix
    """
    mass_tot_inside = (m * inside).sum()

    Sxx = (m * x ** 2 * inside).sum() / mass_tot_inside
    Syy = (m * y ** 2 * inside).sum() / mass_tot_inside
    Szz = (m * z ** 2 * inside).sum() / mass_tot_inside
    Sxy = (m * x * y * inside).sum() / mass_tot_inside
    Sxz = (m * x * z * inside).sum() / mass_tot_inside
    Syz = (m * y * z * inside).sum() / mass_tot_inside

    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def ellipsoidal_shape_particles(x, y, z, m, r, tol=1e-3, maxiter=100, preserve='major', verbose=False):
    """
    Finds the shape of a particle distribution (eigenvalues and eigenvectors of the intertia tensor) by using
    the iterative method in Zemp et al (2011).

    Args:
        x, y, z: (recentered!) position of the particles
        m: particles' masses
        r: initial radius (will be kept as the major semi-axis length
        tol: relative error allowed to the quotient between semiaxes
        maxiter: maximum number of allowed iterations
        preserve: which quantity to preserve when changing the axes (could be 'major', 'intermediate', 'minor' or
                    'volume')

    Returns:
        List of semiaxes lengths and list of eigenvectors.

    """
    inside = x ** 2 + y ** 2 + z ** 2 < r ** 2
    shapetensor = shape_tensor_particles(x, y, z, m, inside)
    S_eigenvalues, S_eigenvectors = tools.diagonalize_ascending(shapetensor)
    lambda_xtilde = S_eigenvalues[0]
    lambda_ytilde = S_eigenvalues[1]
    lambda_ztilde = S_eigenvalues[2]
    u_xtilde = S_eigenvectors[0]
    u_ytilde = S_eigenvectors[1]
    u_ztilde = S_eigenvectors[2]
    axisratio1 = np.sqrt(lambda_ytilde / lambda_ztilde)
    axisratio2 = np.sqrt(lambda_xtilde / lambda_ztilde)
    if preserve == 'major':
        semiax_x = axisratio2 * r
        semiax_y = axisratio1 * r
        semiax_z = r
    elif preserve == 'intermediate':
        semiax_x = axisratio2 / axisratio1 * r
        semiax_y = r
        semiax_z = r / axisratio1
    elif preserve == 'minor':
        semiax_x = r
        semiax_y = axisratio1 / axisratio2 * r
        semiax_z = r / axisratio2
    elif preserve == 'volume':
        semiax_z = r / (axisratio1 * axisratio2) ** (1 / 3)
        semiax_x = axisratio2 * semiax_z
        semiax_y = axisratio1 * semiax_z

    # these will keep track of the ppal axes positions as the thing rotates over and over
    ppal_x = u_xtilde
    ppal_y = u_ytilde
    ppal_z = u_ztilde

    if verbose:
        print('Iteration -1: spherical')
        print('New ratios are', axisratio1, axisratio2)
        print('New semiaxes are', semiax_x, semiax_y, semiax_z)
        print('New eigenvectors are ', ppal_x, ppal_y, ppal_z)

    for i in range(maxiter):
        if verbose:
            print('Iteration {}'.format(i))
        # rotating the particle coordinates
        xprev = x
        yprev = y
        zprev = z
        x = xprev * u_xtilde[0] + yprev * u_xtilde[1] + zprev * u_xtilde[2]
        y = xprev * u_ytilde[0] + yprev * u_ytilde[1] + zprev * u_ytilde[2]
        z = xprev * u_ztilde[0] + yprev * u_ztilde[1] + zprev * u_ztilde[2]
        del xprev, yprev, zprev

        # compute the new 'inside' particles, considering the ellipsoidal shape, keeping the major semiaxis length
        # note that the major semiaxis corresponds to the ztilde component (by construction, see diagonalize_ascending)
        inside = (x / semiax_x) ** 2 + (y / semiax_y) ** 2 + (z / semiax_z) ** 2 < 1

        # keep track of the previous axisratios
        axisratio1_prev = axisratio1
        axisratio2_prev = axisratio2

        # diagonalize the new particle selection
        shapetensor = shape_tensor_particles(x, y, z, m, inside)
        try:
            S_eigenvalues, S_eigenvectors = tools.diagonalize_ascending(shapetensor)
        except np.linalg.LinAlgError:
            print('Shape tensor had {} nans. Operation impossible.'.format(np.isnan(shapetensor).sum()))
            return None, None

        lambda_xtilde = S_eigenvalues[0]
        lambda_ytilde = S_eigenvalues[1]
        lambda_ztilde = S_eigenvalues[2]
        u_xtilde = S_eigenvectors[0]
        u_ytilde = S_eigenvectors[1]
        u_ztilde = S_eigenvectors[2]
        axisratio1 = np.sqrt(lambda_ytilde / lambda_ztilde)
        axisratio2 = np.sqrt(lambda_xtilde / lambda_ztilde)
        if preserve == 'major':
            semiax_x = axisratio2 * r
            semiax_y = axisratio1 * r
            semiax_z = r
        elif preserve == 'intermediate':
            semiax_x = axisratio2 / axisratio1 * r
            semiax_y = r
            semiax_z = r / axisratio1
        elif preserve == 'minor':
            semiax_x = r
            semiax_y = axisratio1 / axisratio2 * r
            semiax_z = r / axisratio2
        elif preserve == 'volume':
            semiax_z = r / (axisratio1 * axisratio2) ** (1 / 3)
            semiax_x = axisratio2 * semiax_z
            semiax_y = axisratio1 * semiax_z
        if verbose:
            print('New ratios are', axisratio1, axisratio2)
            print('New semiaxes are', semiax_x, semiax_y, semiax_z)

        # keep track of the newly rotated vectors
        temp_x = u_xtilde[0] * ppal_x + u_xtilde[1] * ppal_y + u_xtilde[2] * ppal_z
        temp_y = u_ytilde[0] * ppal_x + u_ytilde[1] * ppal_y + u_ytilde[2] * ppal_z
        temp_z = u_ztilde[0] * ppal_x + u_ztilde[1] * ppal_y + u_ztilde[2] * ppal_z
        ppal_x = temp_x
        ppal_y = temp_y
        ppal_z = temp_z
        del temp_x, temp_y, temp_z
        if verbose:
            print('New eigenvectors are ', ppal_x, ppal_y, ppal_z)

        # check for tolerance
        if verbose:
            print('Rel. change is {:.6f} and {:.6f}'.format(axisratio1 / axisratio1_prev - 1,
                                                            axisratio2 / axisratio2_prev - 1))
        if abs(axisratio1 / axisratio1_prev - 1) < tol and abs(axisratio2 / axisratio2_prev - 1) < tol:
            if verbose:
                print('Converged!')
            break

    return [semiax_x, semiax_y, semiax_z], [ppal_x, ppal_y, ppal_z]
