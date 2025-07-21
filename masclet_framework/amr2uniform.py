"""
HALMA-MASCLET FRAMEWORK

Created on Mon Mar 27 2022

@author: ÓSCAR MONLLOR BERBEGAL
"""

import time
import numba
import numpy as np
import sys
import multiprocessing
import tools

# DESCRIPTION
# This function interpolates the AMR field to a uniform grid
# It is faster to clean the patches until l = level and not consider patches with l>level

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" FROM AMR TO UNIFORM GRID WITH NUMBA COMPILATION "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

####################################################
## NUMBA SIGNATURE AND AHEAD OF TIME COMPILATION
####################################################
#[::1]     IS C ORDER
#[:,:,::1] IS C ORDER
#[::1,:,:] IS F ORDER
# signature = numba.float32[:,:,::1](numba.int64, numba.int64, numba.int64, numba.int64, numba.int64, 
#                                    numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], 
#                                    numba.int64, numba.int64, numba.int64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float32[:,:,::1])
# @numba.njit(signature)
@numba.njit(fastmath=True)
def patch_interpolation(field_uniform, level, l, nx, ny, nz, 
                        grid_faces_x, grid_faces_y, grid_faces_z, 
                        grid_centers_x, grid_centers_y, grid_centers_z, 
                        patch_nx, patch_ny, patch_nz, rx, ry, rz, patch_res, patch_field):

    ####################################################################################################################
    # If level == l, we are in the finest level, so we just copy the values to the closest cell
    # If level > l, we are in a coarser level, so we copy the values to the uniform cells CONTAINED in not solaped patch cells
    # If level < l, RAISE AN ERROR, its faster to clean until l = level and not consider patches with l>level
    ####################################################################################################################
    
    #Find the closest cell in the uniform grid
    x0 = rx - patch_res/2 #Center of the left-bottom-front cell
    y0 = ry - patch_res/2
    z0 = rz - patch_res/2
    if l == level:
        #We are in the finest level, so we just copy the values to the closest cell
        #Check if the cell is inside the grid and if it is not solaped with a finer patch
        for i in range(patch_nx):
            x = x0 + i*patch_res #cell center
            if not(x < grid_faces_x[0] or x > grid_faces_x[-1]):
                for j in range(patch_ny):
                    y = y0 + j*patch_res
                    if not(y < grid_faces_y[0] or y > grid_faces_y[-1]):
                        for k in range(patch_nz):
                            z = z0 + k*patch_res
                            if not(z < grid_faces_z[0] or z > grid_faces_z[-1]) and patch_field[i, j, k] != 0.:
                                #FIND THE CLOSEST CELL
                                i_uniform = np.argmin(np.abs(grid_centers_x - x))
                                j_uniform = np.argmin(np.abs(grid_centers_y - y))
                                k_uniform = np.argmin(np.abs(grid_centers_z - z))
                                field_uniform[i_uniform, 
                                              j_uniform, 
                                              k_uniform] = patch_field[i, j, k]

    elif l < level:
        #We are in a coarser level, so we copy the values to the uniform cells CONTAINED in not solaped patch cells
        #Check if the cell is inside the grid (THIS TIME WITH CELL FACES) and if it is not solaped with a finer patch
        for i in range(patch_nx):
            x_left = x0 + i*patch_res - patch_res/2 #cell left face
            x_right = x0 + i*patch_res + patch_res/2 #cell right face
            if not(x_left < grid_faces_x[0] and x_right > grid_faces_x[-1]):
                for j in range(patch_ny):
                    y_left = y0 + j*patch_res - patch_res/2
                    y_right = y0 + j*patch_res + patch_res/2
                    if not(y_left < grid_faces_y[0] and y_right > grid_faces_y[-1]):
                        for k in range(patch_nz):
                            z_left = z0 + k*patch_res - patch_res/2
                            z_right = z0 + k*patch_res + patch_res/2
                            if not(z_left < grid_faces_z[0] and z_right > grid_faces_z[-1]) and patch_field[i, j, k] != 0.:
                                #ASSIGN VALUES
                                i_uniform_left = np.argmin(np.abs(grid_faces_x - x_left))
                                i_uniform_right = min(np.argmin(np.abs(grid_faces_x - x_right)), nx)
                                j_uniform_left = np.argmin(np.abs(grid_faces_y - y_left))
                                j_uniform_right = min(np.argmin(np.abs(grid_faces_y - y_right)), ny)
                                k_uniform_left = np.argmin(np.abs(grid_faces_z - z_left))
                                k_uniform_right =  min(np.argmin(np.abs(grid_faces_z - z_right)), nz)
                                field_uniform[i_uniform_left:i_uniform_right, 
                                              j_uniform_left:j_uniform_right, 
                                              k_uniform_left:k_uniform_right] = patch_field[i, j, k]

    if l > level:
        #RAISE AN ERROR, its faster to clean until l = level and not consider patches with l>level
        raise ValueError('l > level, its faster to clean until l = level and not consider patches with l>level')
    

############################################################################################################
############################################################################################################



            


############################################################################################################
## MAIN FUNCTIONS
############################################################################################################
def iterate_over_patches(which_patches, patch_level, up_to_level, 
                         size, nmax, patchrx, patchry, patchrz, patchnx, patchny, patchnz, 
                         field, nx, ny, nz, grid_faces_x, grid_faces_y, grid_faces_z,
                         grid_centers_x, grid_centers_y, grid_centers_z):
    
    field_uniform = np.zeros((nx, ny, nz), dtype = np.float64)
    for ipatch, patch in enumerate(which_patches):  
        l = patch_level[ipatch]
        if l <= up_to_level:
            patch_res = (size / nmax) / 2**l
            patch_rx = patchrx[ipatch]
            patch_ry = patchry[ipatch]
            patch_rz = patchrz[ipatch]
            patch_nx = patchnx[ipatch]
            patch_ny = patchny[ipatch]
            patch_nz = patchnz[ipatch]
            patch_field = np.ascontiguousarray( field[ipatch] )
            patch_interpolation(field_uniform, up_to_level, l, nx, ny, nz, 
                                grid_faces_x, grid_faces_y, grid_faces_z, 
                                grid_centers_x, grid_centers_y, grid_centers_z, 
                                patch_nx, patch_ny, patch_nz, patch_rx, 
                                patch_ry, patch_rz, patch_res, patch_field)

    return field_uniform


def main(box, up_to_level, nmax, size, 
         npatch, patchnx, patchny, patchnz, 
         patchrx, patchry, patchrz, field,
         ncores = 1, just_that_level = False,
         split_to_box = False, verbose = False):
    """
    box: [xmin, xmax, ymin, ymax, zmin, zmax]
    up_to_level: maximum level to take into account AMR
    nmax: number of cells in the coarsest level
    size: size of the domain
    npatch: number of patches in each level
    patchnx: number of cells in x in each patch
    patchny: number of cells in y in each patch
    patchnz: number of cells in z in each patch
    patchrx: x coordinate of the (l-1) left-bottom-front cell
    patchry: y coordinate of the (l-1) left-bottom-front cell
    patchrz: z coordinate of the (l-1) left-bottom-front cell
    field: AMR field to convert in a uniform field
    ncores: number of cores to use in parallel computation
    just_that_level: if True, it returns the field at that level
    split_to_box: if True, it returns the field sliced to the original box
    verbose: verbosity
    """
    
    # if just_grid = True, it returns only the uniform grid
    if type(field) is not list:
        ValueError('Field must be a list of fields')
    elif len(field) != 1 and len(field) != 3:
        ValueError('Field must be a scalar or a vector')

    #Define uniform grid
    res_coarse = size / nmax

    box_limits = [int((box[0] + size / 2) * nmax / size),
                  int((box[1] + size / 2) * nmax / size),
                  int((box[2] + size / 2) * nmax / size),
                  int((box[3] + size / 2) * nmax / size),
                  int((box[4] + size / 2) * nmax / size),
                  int((box[5] + size / 2) * nmax / size)]
    

    bimin = box_limits[0]
    bimax = box_limits[1]+1
    bjmin = box_limits[2]
    bjmax = box_limits[3]+1
    bkmin = box_limits[4]
    bkmax = box_limits[5]+1

    bxmin = -size / 2 +  bimin      * res_coarse
    bxmax = -size / 2 + (bimax    ) * res_coarse
    bymin = -size / 2 +  bjmin      * res_coarse
    bymax = -size / 2 + (bjmax    ) * res_coarse
    bzmin = -size / 2 +  bkmin      * res_coarse
    bzmax = -size / 2 + (bkmax    ) * res_coarse

    # Interpolation box
    interp_box = [bxmin, bxmax, bymin, bymax, bzmin, bzmax]

    # Boundaries
    xlims = [bxmin, bxmax]
    ylims = [bymin, bymax]
    zlims = [bzmin, bzmax] 
    
    # Number of cells 
    nx = (bimax - bimin) * 2 ** up_to_level
    ny = (bjmax - bjmin) * 2 ** up_to_level
    nz = (bkmax - bkmin) * 2 ** up_to_level

    if verbose:
        print('Number of cells:', nx, ny, nz)

    if nx*ny*nz > 1024**3:
        print('Warning: The number of cells is too high, the interpolation may be slow')

    # Coordinates of the cells
    grid_faces_x = np.linspace(xlims[0], xlims[1], nx+1)
    grid_faces_y = np.linspace(ylims[0], ylims[1], ny+1)
    grid_faces_z = np.linspace(zlims[0], zlims[1], nz+1)
    grid_centers_x = (grid_faces_x[1:] + grid_faces_x[:-1])/2
    grid_centers_y = (grid_faces_y[1:] + grid_faces_y[:-1])/2
    grid_centers_z = (grid_faces_z[1:] + grid_faces_z[:-1])/2

    #SUB-DOMAIN DECOMPOSITION FOR PARALLELIZATION
    #Slice the x-axis in ncores chunks
    if ncores > 1:
        if verbose:
            print('Parallel computation slicing the x-axis in', ncores, 'chunks...')

        t0 = time.time()
        sub_grid_faces_x = np.zeros((ncores, int(nx/ncores)+1), dtype = np.float64)
        for ic in range(ncores):
            ncells_sub = int(nx/ncores)
            sub_grid_faces_x[ic] = grid_faces_x[ic*ncells_sub:ic*ncells_sub+ncells_sub+1]


        sub_grid_centers_x = []
        for ic in range(ncores):
            sub_grid_centers_x.append( (sub_grid_faces_x[ic][1:] + sub_grid_faces_x[ic][:-1])/2 )

        sub_nx = np.zeros(ncores, dtype = np.int64)
        for ic in range(ncores):
            sub_nx[ic] = len(sub_grid_centers_x[ic])
        t1 = time.time()
        if verbose:
            print('Time to slice the x-axis:', t1-t0)

    if verbose:
        print('Interpolating...')

    if ncores == 1:
        patch_level = tools.create_vector_levels(npatch)
        which_patches = tools.which_patches_inside_box(interp_box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax)
        if just_that_level:
            which_patches = [i for i in which_patches if patch_level[i] == up_to_level]
        else:
            which_patches = [i for i in which_patches if patch_level[i] <= up_to_level]

        core_field = [field[i] for i in which_patches]
        core_patch_level = [patch_level[i] for i in which_patches]
        core_patchnx = [patchnx[i] for i in which_patches]
        core_patchny = [patchny[i] for i in which_patches]
        core_patchnz = [patchnz[i] for i in which_patches]
        core_patchrx = [patchrx[i] for i in which_patches]
        core_patchry = [patchry[i] for i in which_patches]
        core_patchrz = [patchrz[i] for i in which_patches]
        
        args = (which_patches, core_patch_level, up_to_level,
                size, nmax, core_patchrx, core_patchry, core_patchrz, core_patchnx, core_patchny, core_patchnz,
                core_field, nx, ny, nz, grid_faces_x, grid_faces_y, grid_faces_z,
                grid_centers_x, grid_centers_y, grid_centers_z)
        
        field_uniform = np.zeros((nx, ny, nz), dtype = np.float32)
        t0 = time.time()
        field_uniform = iterate_over_patches(*args)
        tf = time.time()

        if verbose:
            print('Time to interpolate ( serial ):', tf-t0)
        
    elif ncores > 1:
        data_sub = []
        t0 = time.time()
        for ic in range(ncores):
            patch_level = tools.create_vector_levels(npatch)
            core_box = [sub_grid_faces_x[ic][0], sub_grid_faces_x[ic][-1], ylims[0], ylims[1], zlims[0], zlims[1]]
            which_patches = tools.which_patches_inside_box(core_box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax)
            if just_that_level:
                which_patches = [i for i in which_patches if patch_level[i] == up_to_level]
            else:
                which_patches = [i for i in which_patches if patch_level[i] <= up_to_level]

            core_field = [field[i] for i in which_patches]
            core_patch_level = [patch_level[i] for i in which_patches]
            core_patchnx = [patchnx[i] for i in which_patches]
            core_patchny = [patchny[i] for i in which_patches]
            core_patchnz = [patchnz[i] for i in which_patches]
            core_patchrx = [patchrx[i] for i in which_patches]
            core_patchry = [patchry[i] for i in which_patches]
            core_patchrz = [patchrz[i] for i in which_patches]

            this_core_args = (which_patches, core_patch_level, up_to_level,
                            size, nmax, core_patchrx, core_patchry, core_patchrz, core_patchnx, core_patchny, core_patchnz,
                            core_field, sub_nx[ic], ny, nz, sub_grid_faces_x[ic], grid_faces_y, grid_faces_z,
                            sub_grid_centers_x[ic], grid_centers_y, grid_centers_z)
                            
            data_sub.append(this_core_args)
        t1 = time.time()
        if verbose:
            print('Preparing parallelisation:', t1-t0)


        t0 = time.time()
        with multiprocessing.get_context('fork').Pool(ncores) as pool:
            results = pool.starmap(iterate_over_patches, data_sub)

        #Stack the results along the x-axis
        field_uniform = results[0]
        for ic in range(1, ncores):
            field_uniform = np.concatenate((field_uniform, results[ic]), axis = 0)

        tf = time.time()

        if verbose:
            print('Time to interpolate ( parallel ):', tf-t0)

    if verbose:
        print('Done')

    if split_to_box:
        
        #Slice the field to original box
        x0 = box[0]
        x1 = box[1]
        y0 = box[2]
        y1 = box[3]
        z0 = box[4]
        z1 = box[5]

        #Find the closest center in the uniform grid
        i0 = np.argmin(np.abs(grid_centers_x - x0))
        i1 = np.argmin(np.abs(grid_centers_x - x1)) + 1
        j0 = np.argmin(np.abs(grid_centers_y - y0))
        j1 = np.argmin(np.abs(grid_centers_y - y1)) + 1
        k0 = np.argmin(np.abs(grid_centers_z - z0))
        k1 = np.argmin(np.abs(grid_centers_z - z1)) + 1
        

        # Check if the box is a cube
        dx = round(x1-x0, 6)
        dy = round(y1-y0, 6)
        dz = round(z1-z0, 6)
        box_is_cubical = (dx == dy) and (dy == dz)
        if box_is_cubical:
            #If it is a cube, we take the same number of cells in each direction
            nmax_cell  = max(i1-i0, j1-j0, k1-k0)
            i1 = i0 + nmax_cell
            j1 = j0 + nmax_cell
            k1 = k0 + nmax_cell

        grid_centers_x = grid_centers_x[i0:i1]
        grid_centers_y = grid_centers_y[j0:j1]
        grid_centers_z = grid_centers_z[k0:k1]
        grid_faces_x = grid_faces_x[i0:i1+1]
        grid_faces_y = grid_faces_y[j0:j1+1]
        grid_faces_z = grid_faces_z[k0:k1+1]

        field_uniform = field_uniform[i0:i1, j0:j1, k0:k1]

    return field_uniform, grid_faces_x, grid_faces_y, grid_faces_z
    
############################################################################################################
############################################################################################################