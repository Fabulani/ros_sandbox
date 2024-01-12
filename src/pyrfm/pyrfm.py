import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # progress bar



# READ ROSBAG
# DUMP DATA TO rays.rays:
# rays_struct = np.dtype(
#     [
#         ('sx', np.float32),
#         ('sy', np.float32),
#         ('ex', np.float32),
#         ('ey', np.float32),
#         ('r' , np.float32),
#         ('th', np.float32),
#         ('i',  np.int32),
#         ('idx',  np.int32),
#         ('ts', np.float64), 
#       ])
# - sx,sy : the start of the ray (i.e. lidar sensor center) in the map coordinates 
# - ex,ey : the endpoint of the ray
# - r, th, i : original range, angle, and intensity reported by the sensor \[th, i are unused\]
# - idx : the identifier of the sensor the ray is from \[unused\]
# - ts : Unix timestamp of the ray \[unused\]


# Change this to the ray file you're using!
rayfile = "rays.rays"
data_dir = "../../data/"
raypath = data_dir + rayfile



# ------------ HYPERPARAMETERS
th_spacing = 2*np.pi/360/4 # 1/4 degree spacing from the sensor
th_bins = int(np.round(2*np.pi / th_spacing)) 
W, H = 800,1200


bias = 0
wide_threshold = 30*np.pi/180 / th_spacing
epsl = 1.

## Derived constants
# w_HIT = 3 # Reduce to 1 if known no motion
# w_MISS = 1 
# REF_BLOCK_THRESH = w_MISS/(w_MISS+w_HIT) # the theoretical way. Fraction of hits to total that means we have a reflective voxel
REF_BLOCK_THRESH = 0.3 # this is a good 

# ---------------- Large persistent memory blocks. 
HIT = np.zeros((W,H,th_bins),'int16')
MISS = np.zeros((W,H,th_bins),'int16')
RFM = np.zeros((W,H,th_bins),'int8') # -1=TRANSPARENT, 0=UNK, 1=REFLECT
RFM_mr = np.zeros((W,H,th_bins),'int8')
WIDE = np.zeros((W,H,th_bins),'bool')
NARROW = np.zeros((W,H,th_bins),'bool')
prerender = np.zeros((W,H),'bool')
refl_cache = np.zeros((W,H),'int16')
trans_cache = np.zeros((W,H),'int16')
OCC = np.zeros((W,H),'int8')

vox_refl = np.zeros((W,H,th_bins),'float32')  
CLASSIFIED_RFM = np.zeros((W,H,th_bins),'bool')  
countvis = np.zeros((W,H),'uint16')
counttrans = np.zeros((W,H),'uint16')
countratio = np.zeros((W,H),'float32')


# ------------ LOAD RAYS
rays_struct = np.dtype(
    [
        ('sx', np.float32),
        ('sy', np.float32),
        ('ex', np.float32),
        ('ey', np.float32),
        ('r' , np.float32),
        ('th', np.float32),
        ('i',  np.int32),
        ('idx',  np.int32),
        ('ts', np.float64), 
      ])

# rays0 = np.memmap(rayfile,rays_struct, 'r',1) # if you want, you can memory map the file.
rays0 = np.fromfile(raypath, rays_struct)
rays0 = rays0[:int(len(rays0)//1081*1081)]

##########

## Find the max extent of the map and transforms

keep=(rays0['r']<10000)*(rays0['r']<10000)
rays=rays0[keep==1]

## Figure out rasterization
# Floor of float coords is integer coords
# Images are y flipped

def homp(T,pts):
    tmppts = pts @ T[:-1,:-1].T + T[:-1,-1]
    denom = pts @ T[-1:,:-1].T + T[-1,-1]
    return tmppts/denom


CELLS_PER_M = 20.0
print("Cell width:", 1/CELLS_PER_M)

formatter = {'float':lambda x:np.format_float_positional(x,precision=2,fractional=True,trim='-',pad_left=8,pad_right=2)}

def get_T_px_f_with_extent(rays, debug=False):
    # The corners in world space
    f_corns = np.array(
        [[np.min(rays['ex']),np.min(rays['ey'])],
         [np.max(rays['ex']),np.max(rays['ey'])]])
    T_pxc_f = np.array(
        [[1*CELLS_PER_M, 0             , 0],
         [0            , -1*CELLS_PER_M, 0],
         [0            , 0             , 1]])
    pxc_corns=homp(T_pxc_f,f_corns)
    pxc_topleft = np.array([  np.floor(np.min(pxc_corns[:,0])), np.floor(np.min(pxc_corns[:,1]))  ]) #inclusive
    pxc_btmright= np.array([  np.floor(np.max(pxc_corns[:,0])), np.floor(np.max(pxc_corns[:,1]))  ]) + 1 #exclusive
    pxc_extent = pxc_btmright-pxc_topleft

    T_px_f = T_pxc_f.copy()
    T_px_f[:-1,-1] = -pxc_topleft

    px_corns=homp(T_px_f,f_corns)
    px_topleft = np.array([  np.floor(np.min(px_corns[:,0])), np.floor(np.min(px_corns[:,1]))  ]) #inclusive
    px_btmright= np.array([  np.floor(np.max(px_corns[:,0])), np.floor(np.max(px_corns[:,1]))  ]) + 1 #exclusive
    px_extent = px_btmright-px_topleft

    assert (np.all(pxc_extent == px_extent))

    if debug:
        print("fcorns\n",f_corns)

        for var in ['pxc_corns',
                    'pxc_topleft',
                    'pxc_btmright',
                    'pxc_extent',
                    'px_corns',
                    'px_topleft',
                    'px_btmright',
                    'px_extent',
                    ]:
            print(var+'\n', eval(var))
    return T_px_f, px_extent.astype('int64')

keep_for_window=(rays0['r']>0)*(rays0['r']<10)
with np.printoptions(formatter=formatter):
    T_px_f, px_extent = get_T_px_f_with_extent(rays[keep_for_window], debug=True)



# --------------- CONVERTR RAYS TO GRID COORDS
# # Debug show input rays
# %matplotlib widget
# plt.plot(rays0['ex'],rays0['ey'],'b,')


# # If you need to edit the rays, but keep extents, do it here
# # Example: Slice by endpoint
# rays0 = rays0[rays0['ex']>50 ]
# rays0 = rays0[rays0['ex']<55 ]
# rays0 = rays0[rays0['ey']<23 ]
# print(len(rays0))

# # Example 2: Slice away a certain number of rays
# substart = 0
# subend = int(2.0e6)
# rays0 = rays0[substart:subend]
# rays0 = rays0[:int(len(rays0)//1081*1081)] # Trick to cut off ragged scans to make blocks of length 1081

# # Example 3: Slice by time
# rays0 = rays0[rays0['ts']< 1479396535.3080368]
rays = rays0.copy()

print("# Rays to map:", len(rays0))
plt.plot(rays0['ex'],rays0['ey'],'k,')
plt.plot(rays0['sx'],rays0['sy'],'c,')
plt.savefig("out/rays_to_map.png")
print("Time range to map:", np.min(rays0['ts']),np.max(rays0['ts']))

# convert to [start_cell_x,start_cell_y, d_cell, th] format
f_spt = np.vstack((rays['sx'].ravel(),rays['sy'].ravel())).T.copy()
f_ept = np.vstack((rays['ex'].ravel(),rays['ey'].ravel())).T.copy()
px_spt = homp(T_px_f, f_spt)
px_ept = homp(T_px_f, f_ept)
px_d = np.sqrt(np.sum((px_ept-px_spt)**2,axis=1))
px_th = np.mod(np.arctan2((px_ept-px_spt)[:,1], (px_ept-px_spt)[:,0]),2*np.pi)
def get_raster_coords(rays, T_px_f):
    # implicit inputs: T_px_f
    f_ept = np.vstack((rays['ex'].ravel(),rays['ey'].ravel())).T.copy()
    px_coords = homp(T_px_f, f_ept)
    px_coords = px_coords.reshape(rays['ex'].shape+(2,))
    return px_coords

th_spacing = 2*np.pi/360/4

outrays = np.vstack([px_spt.T,px_d,px_th/th_spacing])
notref = np.ones_like(outrays[0], dtype=bool)

# #
# plt.plot(px_ept[:,0],px_ept[:,1],'b,')
# plt.plot(px_ept[10000:101081,0],px_ept[10000:101081,1],'r,')