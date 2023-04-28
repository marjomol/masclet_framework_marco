"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

profiles module
Several functions to compute directional profiles of gridded data.
Created by David Vallés
"""

from numba import jit
import numpy as np
from masclet_framework import tools
from tqdm import tqdm

@jit(nopython=True, fastmath=True)
def locate_point(x,y,z,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,nl,buf=1):
    """
    Given a point (x,y,z) and the patch structure, returns the patch and cell
    where the point is located.

    Args:
        - x,y,z: coordinates of the point
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: cells at base level
        - nl: maximum AMR level to be considered
        - buf: number of cells to be ignored in the border of each patch

    Returns:
        - f_patch: patch where the point is located
        - ix,jy,kz: cell where the point is located
    """
    f_patch=0
    lev_patch=0
    for l in range(nl,0,-1):
        #print(l)
        low1=npatch[0:l].sum()+1
        low2=npatch[0:l+1].sum()
        dxpa=size/nmax/2**l
        for ipatch in range(low1,low2+1):
            x1=patchrx[ipatch]+(buf-1)*dxpa
            x2=patchrx[ipatch]+(patchnx[ipatch]-1-buf)*dxpa
            if x1>x or x2<x:
                continue

            y1=patchry[ipatch]+(buf-1)*dxpa
            y2=patchry[ipatch]+(patchny[ipatch]-1-buf)*dxpa
            #print(y1,y2)
            if y1>y or y2<y:
                continue
            z1=patchrz[ipatch]+(buf-1)*dxpa
            z2=patchrz[ipatch]+(patchnz[ipatch]-1-buf)*dxpa
            if z1>z or z2<z:
                continue
            
            f_patch=ipatch
            lev_patch=l
            #print('a')
            break
        
        if f_patch>0:
            break
    
    dxpa=size/nmax/2**lev_patch
    x1=patchrx[f_patch]-dxpa
    y1=patchry[f_patch]-dxpa
    z1=patchrz[f_patch]-dxpa
    
    ix=int((x-x1)/dxpa)
    jy=int((y-y1)/dxpa)
    kz=int((z-z1)/dxpa)
    
    ix=int((x-x1)/dxpa-.5)
    jy=int((y-y1)/dxpa-.5)
    kz=int((z-z1)/dxpa-.5)
    
    return f_patch,ix,jy,kz

def dir_profile(field, cx,cy,cz,
                npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,
                binsphi=None, binscostheta=None, binsr=None, rmin=None, rmax=None, dex_rbins=None, delta_rbins=None,
                interpolate=False):
    """
    Computes a directional profile of a given field, centered in a given point (cx,cy,cz).
    There are several ways to specify the directions along which the profile is computed:
        - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
        - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
    Likewise, there are several ways to specify the radial bins:
        - binsr: numpy vector specifying the radial bin edges
        - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
        - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
    The profile can be computed by nearest neighbour interpolation (interpolate=False) or by averaging the values of the cells in each bin (interpolate=True).

    Args:
        - field: field to be profiled
        - cx,cy,cz: coordinates of the center of the profile
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: cells at base level
        One and only one of these pairs of arguments must be specified:
            - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
            - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
        One and only one of these sets of arguments must be specified:
            - binsr: numpy vector specifying the radial bins
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
        - interpolate: if True, the profile is computed by averaging the values of the cells in each bin. If False, the profile is computed by nearest neighbour interpolation.

    Returns:
        - dirprof: directional profile of the field
        - rrr: radial bins
        - vec_costheta: cos(theta) bins
        - vec_phi: phi bins
    """
    
    # Check phi bins are properly specified
    if type(binsphi) is np.ndarray or type(binsphi) is list:
        Nphi=len(binsphi)
        vec_phi=binsphi
    elif type(binsphi) is int:
        Nphi=binsphi
        vec_phi=np.linspace(-np.pi + np.pi/Nphi,np.pi -np.pi/Nphi,Nphi)
    else:
        raise ValueError('Wrong specification of binsphi')
        
    # Check theta bins are properly specified
    if type(binscostheta) is np.ndarray or type(binscostheta) is list:
        Ncostheta=len(binscostheta)
        vec_costheta=binscostheta
    elif type(binscostheta) is int:
        Ncostheta=binscostheta
        vec_costheta=np.linspace(-1+1/Ncostheta,1-1/Ncostheta,Ncostheta)
    else:
        raise ValueError('Wrong specification of binscostheta')  
        
    # Check r bins are properly specified
    if type(binsr) is np.ndarray or type(binsr) is list:
        num_bins=len(binsr)
        rrr=binsr
    elif (rmin is not None) and (rmax is not None) and ((dex_rbins is not None) or (delta_rbins is not None)) and ((dex_rbins is None) or (delta_rbins is None)):
        if dex_rbins is not None:
            num_bins=int(np.log10(rmax/rmin)/dex_rbins/2)*2+1 # guarantee it is odd
            rrr = np.logspace(np.log10(rmin),np.log10(rmax),num_bins)
        else:
            num_bins=int((rmax-rmin)/delta_rbins/2)*2+1 # guarantee it is odd
            rrr = np.linspace(rmin,rmax,num_bins)
    else:
        raise ValueError('Wrong specification of binsr') 
        
        

    levels=tools.create_vector_levels(npatch)
    nl=levels.max()
        
    drrr=np.concatenate([[rrr[1]-rrr[0]], np.diff(rrr)])
    lev_integral = np.clip(np.log2((size/nmax)/drrr).astype('int32'),0,nl)#+1
    del drrr
    
    dir_profiles = np.zeros((Ncostheta,Nphi,num_bins))

    if interpolate:
        for itheta,costheta in tqdm(enumerate(vec_costheta),total=len(vec_costheta)):
            for jphi,phi in enumerate(vec_phi):
                xxx=cx+rrr*np.sqrt(1-costheta**2)*np.cos(phi)
                yyy=cy+rrr*np.sqrt(1-costheta**2)*np.sin(phi)
                zzz=cz+rrr*costheta

                for kbin,(xi,yi,zi,li) in enumerate(zip(xxx,yyy,zzz,lev_integral)):
                    ip,i,j,k=locate_point(xi,yi,zi,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,li)
                    ll=levels[ip]
                    dxx=(xi-(patchrx[ip]+(i-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    dyy=(yi-(patchry[ip]+(j-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    dzz=(zi-(patchrz[ip]+(k-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    #assert 0 <= dxx <= 1
                    #assert 0 <= dyy <= 1
                    #assert 0 <= dzz <= 1
                    dir_profiles[itheta,jphi,kbin]=field[ip][i,j,k]      *(1-dxx)*(1-dyy)*(1-dzz) \
                                                 + field[ip][i,j,k+1]    *(1-dxx)*(1-dyy)*  dzz   \
                                                 + field[ip][i,j+1,k]    *(1-dxx)*  dyy  *(1-dzz) \
                                                 + field[ip][i,j+1,k+1]  *(1-dxx)*  dyy  *  dzz   \
                                                 + field[ip][i+1,j,k]    *  dxx  *(1-dyy)*(1-dzz) \
                                                 + field[ip][i+1,j,k+1]  *  dxx  *(1-dyy)*  dzz   \
                                                 + field[ip][i+1,j+1,k]  *  dxx  *  dyy  *(1-dzz) \
                                                 + field[ip][i+1,j+1,k+1]*  dxx  *  dyy  *  dzz  
    else:
        for itheta,costheta in tqdm(enumerate(vec_costheta),total=len(vec_costheta)):
            for jphi,phi in enumerate(vec_phi):
                xxx=cx+rrr*np.sqrt(1-costheta**2)*np.cos(phi)
                yyy=cy+rrr*np.sqrt(1-costheta**2)*np.sin(phi)
                zzz=cz+rrr*costheta

                for kbin,(xi,yi,zi,li) in enumerate(zip(xxx,yyy,zzz,lev_integral)):
                    ip,i,j,k=locate_point(xi,yi,zi,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,li)
                    dir_profiles[itheta,jphi,kbin]=field[ip][i,j,k]
                    
    return dir_profiles, rrr, vec_costheta, vec_phi



