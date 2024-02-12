import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # progress bar



# READ ROSBAG
# DUMP DATA TO rays.rays:
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
# - sx,sy : the start of the ray (i.e. lidar sensor center) in the map coordinates 
# - ex,ey : the endpoint of the ray
# - r, th, i : original range, angle, and intensity reported by the sensor \[th, i are unused\]
# - idx : the identifier of the sensor the ray is from \[unused\]
# - ts : Unix timestamp of the ray \[unused\]


# Change this to the ray file you're using!
rayfile = "standing_still-once.rays"
data_dir = "../../data/"
data_dir = "./out/"
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
# 
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


# ------------ BASIC RFM SECTION ---------------- #

loclip = np.array([0,0,0]).reshape(3,1)
hiclip = (np.array(HIT.shape)-1).reshape(3,1)
def quantize(coords):# 3xN
    return np.clip(np.floor(coords),loclip,hiclip).astype('int')

def quantize4(coords): # 4 pt xy antialiasing samples # 3xN
    coords = np.array(coords).T  # Nx3
    aa = np.array(((-0.5,-0.5,0.0), (-0.5,0.5,0.0), 
                  ( 0.5,-0.5,0.0), ( 0.5,0.5,0.0) )) # 4x3
    coords4 = aa + coords[:,np.newaxis,:] # N,4,3
    coords4= coords4.reshape(-1,3).T
    return np.floor(np.clip(coords4,loclip,hiclip)).astype('int')

def RFM_update_cell(x,y,th):
    # Unused in batch mode! Would update floodfill connected components in non-batch
    pass
    
# Define the accumulate operator, that renders rays
def accumulate(rays, notref):
    
    

    # ray has a starting point, distance before it returned, and direction
    [x_start, y_start, d, th] = rays
    [slope_x, slope_y] = [np.cos(th*th_spacing), np.sin(th*th_spacing)]
    
    # prerender lets us stop rendering a ray that would go through a HIT
    prerender[:,:]=0
    xyth = quantize4((x_start[notref] + d[notref]*slope_x[notref], y_start[notref] + d[notref]*slope_y[notref], th[notref]))
    prerender[xyth[0],xyth[1]]=1
    

    # ray has a starting point, distance before it returned, and direction
    [x_start, y_start, d, th] = rays
    [slope_x, slope_y] = [np.cos(th*th_spacing), np.sin(th*th_spacing)]
    
    
    # Render all MISSES in parallel. Rays stop drawing when past d-epsl from the start
    live=np.ones_like(d)
    r=np.full_like(d,0)
    for r0 in range(0,200):
        r[:]=r0
        live[r0>=d-epsl]=0
        r[live==0]=100000
        xyth = quantize((x_start + r*slope_x, y_start + r*slope_y, th))
        live*=(prerender[xyth[0],xyth[1]]==0)
        MISS[xyth[0],xyth[1],xyth[2]]+=1
        RFM_update_cell(xyth[0],xyth[1],xyth[2])
    xyth = quantize((x_start[notref] + d[notref]*slope_x[notref], y_start[notref] + d[notref]*slope_y[notref], th[notref]))
    HIT[xyth[0],xyth[1],xyth[2]]+=1
    RFM_update_cell(xyth[0], xyth[1], xyth[2])


# The basic RFM doer. 
def doRFM():
    print("Building RFM")
    MISS[...] = 0
    HIT[...] = 0
    arays = outrays.copy()

    ls=0
    keep=(rays0['r']<10)*(rays0['r']>0)
    CHUNK_SIZE = 1081
    for s in tqdm(range(CHUNK_SIZE,arays.shape[1]+1,CHUNK_SIZE)):
            tmprays=arays[:,ls:s][:,keep[ls:s]]
            nrtemp=notref[ls:s][keep[ls:s]]
            accumulate(tmprays, nrtemp)
            ls=s
    plt.imshow(np.sum(MISS,axis=2))
    # Save the image locally
    plt.savefig('out/building_rfm.png')
    assert np.max(MISS)>0
    vox_refl[...] = HIT/(np.float32(0.1)+HIT+MISS) # classify 
    CLASSIFIED_RFM[...] = vox_refl > REF_BLOCK_THRESH
    countvis[...] = np.sum(CLASSIFIED_RFM, axis=2).astype(np.uint16)
    counttrans[...] = np.sum((CLASSIFIED_RFM==0)&(MISS>0), axis=2).astype(np.uint16)
    countratio[...] = np.float32(1.0*countvis/(countvis+counttrans+0.0000000001))
    refl_cache[...] = (countratio > 0.5) | (countvis > 12)  # More reflective than not or a suspiciously wide range of sightings




# Utility that unions two sets of labelled data. Used to turn normal flood fill into circular floodfill around theta.
def unify_labels(rfmcomps, rfmcomps2):
    # unify labels is mostly a wrapper around fuse, but with the optimization that we only try to 
    # fuse nonzero labels, since we know they're the same
    def fuse(A,B):
        # Assumes A and B are label matricies with different labels and we need to fuse them
        # Any repeated label between the two corresponds
        # Assumes that the labels go from 0 to max(max(A),max(B))
        # Also assumes all labels in A are <= to the ones in B
        assert(np.all(A<=B))
  
        lookup = np.arange(np.maximum(np.max(A),np.max(B))+1, dtype=np.int32)
        np.minimum.at(lookup,B,A) # Bs now lookup the lowest A they connect to, As loopback

        B=lookup[B] #Now all Bs are <= A

        np.minimum.at(lookup,A,B)
        #As lookup the smallest B they connect to, Bs lookup the smallest A or better
        B=lookup[lookup[B]] 

        extr=(A!=B) 
        remA=A[extr]
        remB=B[extr]
        if len(remA)==0:
            
            return B
        recur = fuse(remB,remA)
        B[extr] = recur
        return B
    
    nz=fuse(rfmcomps[rfmcomps!=0].ravel(),rfmcomps2[rfmcomps2!=0].ravel())
    rfmcomps[rfmcomps!=0]=nz
    return rfmcomps

# fuse(np.array([1,2,2,3,3,4,4]),np.array([5,6,7,7,8,8,9]))



# ---------------- CLASSIFY REFLECTION RAYS SECTION ------------ #


# classify reflectionness of a ray and clip to a given penetration depth
PEN_DEPTH = np.ceil(0.101*CELLS_PER_M)
def classify(rays, pen_depth=None):
    rays=rays.copy()
    if pen_depth is None:
        pen_depth = PEN_DEPTH
    HACK_FACTOR = 2 # ensure that detected reflections still count as seen through for this distance
#     for ray in rays:
    # ray has a starting point, distance before it returned, and direction
    [x_start, y_start, d, th] = rays
    [slope_x, slope_y] = [np.cos(th*th_spacing), np.sin(th*th_spacing)]
    
    
    notref=np.ones_like(d,dtype=bool)
    r=np.full_like(d,0)
    for r0 in tqdm(range(0,200)):
#         print(r0,end=', ')
        r[:]=r0
        xyth = quantize((x_start + r*slope_x, y_start + r*slope_y, th))
        
#         if r0>1
        stopped=(refl_cache[xyth[0],xyth[1]]!=0)
        notrefish = r0 > d[stopped]-pen_depth
        notref[stopped] *= notrefish
#         print(r0+pen_depth)
       
        d[stopped]=np.minimum(d[stopped],r0+pen_depth+HACK_FACTOR) 
        
    rays = np.vstack([x_start, y_start, d, th])
    return rays, notref


# The returned rays now have distances reflecting how far to render them. 
# If the distance is less than the original, we want to not render the HIT when we rerender
# "notref" keeps track of which rays should render their endpoint

# This operation is ~1/3 the work of the normal RFM render, but I haven't optimized it, so it 
# runs much slower when the ray list is too big fo the cache


# ----------- MOTION REMOVAL --------- #


# This function produces a label array rfmcomps, that labels everything that is connected in RFM space, accounting for the wraparound of theta
# It works by labelling connected components, rolling Pi around, relabelling, and then merging the resulting label set

from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure

def label_connected_components():
    global rfmcomps
    vox_refl[...] = HIT/(np.float32(0.1)+HIT+MISS)
    structure=generate_binary_structure(3,3)
    rfmcomps, count = label(vox_refl>REF_BLOCK_THRESH, structure)

    roll_vox_refl=np.roll(vox_refl, int(vox_refl.shape[2]//2), axis=2)
    rfmcomps2, count = label(roll_vox_refl>REF_BLOCK_THRESH, structure)
    del roll_vox_refl
    offset=np.max(rfmcomps)+1
    rfmcomps2=np.roll(rfmcomps2, -int(rfmcomps2.shape[2]//2), axis=2)+offset
    rfmcomps2[rfmcomps2==offset]=0
    rfmcomps = unify_labels(rfmcomps, rfmcomps2)
    del rfmcomps2
    return rfmcomps


# This cell removes motion from RFM, given a label array for connected components
def selectH():
    global HIGHLY_VISIBLE
    global rfmcomps
    global selection_color
    rfmcomps = label_connected_components()
    HIGHLY_VISIBLE = (countvis>wide_threshold)
    selection_color = np.max(rfmcomps)+1

    plt.imshow(HIGHLY_VISIBLE.T)
    plt.savefig('out/selectH_highly_visible.png')
    

    # Build an array that has the selection_color on highly visible items, and the old color on everythng else
    rfmcomps2 = (rfmcomps > 0)*HIGHLY_VISIBLE[:,:,np.newaxis]*selection_color
    rfmcomps2[rfmcomps2==0]=rfmcomps[rfmcomps2==0]

    # # Debug: check the selelection array
    # to_show=np.nonzero(rfmcomps2!=rfmcomps2[-3,-3,-3])
    # per=np.random.permutation(int(np.max(rfmcomps2))+1)
    # %gui qt
    # import mayavi.mlab
    # mayavi.mlab.points3d(to_show[0], to_show[1], to_show[2],per[rfmcomps2[to_show].astype(int)], colormap='spectral',mode= 'point')

    # Flood from the seeds
    rfmcomps = unify_labels(rfmcomps, rfmcomps2)



# ----- RUN EVERYTHING SECTION ----- #
    
keeprays=(rays0['r']<10)*(rays0['r']>0) # needed in ablations
doRFM()

plt.imshow(refl_cache.T)
plt.title('Cells that stop rays for reflectance calculations')
plt.savefig('out/cells_that_stop_rays_for_reflectance_calculations.png')


selectH()

# Selected Cells now have the highest label id.
newselcolor=np.max(rfmcomps[HIGHLY_VISIBLE,:])
selected = np.sum(rfmcomps==newselcolor,axis=2)

# We keep all locations that are selected or have a HIT with no MISS 
nevermissed = ((np.sum(HIT,axis=2)>0)&(np.sum(MISS,axis=2)==0))
allkept = 0.5*nevermissed+selected

# Use this for reflection removal calculations, rather than the one acumulated in the basic RFM
# Skiping this lets you ablate how the motion removal affects reflection removal
refl_cache[...] = (allkept!=0)

# Display the result summary so far

plt.imshow(allkept.T>0)

# Visualize after flood fill

print(rfmcomps.shape)

plt.imshow(np.clip(selected.T,0,15))



# ----------- TIME TO REMOVE REFLECTIONS AND DO ALL OVER AGAIN


keeprays=(rays0['r']<10)*(rays0['r']>0)

# Reflection removal
arays,notref = classify(outrays)

# Repeat RFM building
doRFM()

# Repeat motion removal
selectH()
# Now we're done! everything after this is just display







# Selected Cells now have the highest label id.
newselcolor=np.max(rfmcomps[HIGHLY_VISIBLE,:])
selected = np.sum(rfmcomps==newselcolor,axis=2)

# We keep all locations that are selected or have a HIT with no MISS 
nevermissed = ((np.sum(HIT,axis=2)>0)&(np.sum(MISS,axis=2)==0))
allkept = 0.5*nevermissed+selected

# Use this for reflection removal calculations, rather than the one acumulated in the basic RFM
# Skiping this lets you ablate how the motion removal affects reflection removal
refl_cache[...] = (allkept!=0)

# Display the result summary so far

plt.imshow(allkept.T>0)




# Build Ratio Metric for Occupancy 
countvis = np.sum(CLASSIFIED_RFM, axis=2).astype(np.uint16)
counttrans = np.sum((CLASSIFIED_RFM==0)&(MISS>0), axis=2).astype(np.uint16)
countratio = np.float32(1.0*countvis/(countvis+counttrans+0.0000000001))

markedrfc = (rfmcomps>0)*countratio[:,:,np.newaxis]







import sys

infofile = str(round(rays0['ts'][0]*1e6))+"_"+str(round(rays0['ts'][-1]*1e6))+".info"
print("Saving to:",infofile)
print()

original_stdout = sys.stdout # Save a reference to the original standard output

with open(infofile, 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("# Transform between map and pixels (opengl flooring convention):")
    print("T_px_f=",T_px_f.tolist())
    print("# Image info")
    print("W,H,th_bins=",(W,H,th_bins))
    print("# Which rays were used:")
    print("start_time=", str(np.min(rays0['ts'])))
    print("end_time=", str(np.max(rays0['ts'])))
    print("# Base dataset:")
    print("rayfile=",str(rayfile))
    sys.stdout = original_stdout 
with open(infofile, 'r') as f:
    print(f.read())
import cv2

def writeim(name, IM):
    # storage format is 16 bit unsigned
    layers=[IM[:,:,layer] for layer in range(IM.shape[2])]
    batches=[np.hstack(layers[off:off+40]) for off in range(0,IM.shape[2],40)]
    rearranged = np.vstack(batches)
    cv2.imwrite(name,rearranged.astype(np.uint16))
writeim("MISS.png", MISS)
writeim("HIT.png", HIT)
cv2.imwrite("refl_cache.png",refl_cache)
cv2.imwrite("allkept.png" , ((allkept>0)*255).astype(np.uint8))
countvis = np.sum(CLASSIFIED_RFM, axis=2).astype(np.uint16)
counttrans = np.sum((CLASSIFIED_RFM==0)&(MISS>0), axis=2).astype(np.uint16)
cv2.imwrite("countvis.png" , countvis)
cv2.imwrite("counttrans.png" , counttrans)
cv2.imwrite("countratio.png" , (countvis*1.0/(counttrans+countvis)*255).astype('uint8'))

