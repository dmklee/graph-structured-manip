import numpy as np

class NoValidStructureState(Exception):
    def __init__(self, message='', img=None):
        if img is not None:
            import matplotlib.pyplot as plt
            import time
            plt.figure()
            plt.imshow(img)
            plt.title(message)
            plt.savefig(str(time.time())+'.png')
            plt.close()
        super().__init__()

class NoValidPositionException(Exception):
    pass

def normalize_img(img, irange):
    """Converts a image of dtype float to np.uint8 by remapping to irange

    :inputs:
        :img: np array
        :irange: length 2 array of low and high values for remapping
    :return:
        :norm_img: np array with dtype=np.uint8
    """
    norm_img = ( 255*((img-irange[0])/(irange[1]-irange[0])).clip(min=0,max=1) ).astype(np.uint8)
    return norm_img

def rotate_SO2(points, angle):
    """Rotates 2d points by prescribed angle

    :inputs:
        :points: np array (N,2)
        :angle: float variable in radians

    :return: rotated points np array (N,2)
    """
    if isinstance(angle, np.ndarray):
        R = np.vstack((np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)))
        R = R.T.reshape(-1, 2, 2)
    else:
        R = np.array(((np.cos(angle),-np.sin(angle)),
                      (np.sin(angle), np.cos(angle))))
    return np.dot(points, R.T)

def getAABB(footprint, length, width, padding=0):
    """Get axis-aligned bounding box.

    :inputs:
        :footprint: np array (x,y,theta)
        :length: size in x direction
        :width: size in y direction
        :padding: padding added around each object side, this gives some spacing
        so objects arent right next to each other
    :returns:
        :AABB: np array ((Ax,Ay),(Bx,By))
    """
    points = 0.5*np.array((( length+padding, width+padding),
                           (-length-padding, width+padding),
                           ( length+padding, -width-padding),
                           (-length-padding, -width-padding)))
    rot_points = rotate_SO2(points, footprint[2]) + footprint[:2]
    AABB = np.stack((rot_points.min(0),rot_points.max(0)))
    return AABB

def vec_getAABB(footprint, length, width, padding=0):
    """Get axis-aligned bounding box.

    :inputs:
        :footprint: np array of size Nx3 [(x,y,theta),...]
        :length: size in x direction
        :width: size in y direction
        :padding: padding added around each object side, this gives some spacing
        so objects arent right next to each other
    :returns:
        :AABB: np array Nx2x2 [ ((Ax,Ay),(Bx,By)), ...]
    """
    rel_corners = 0.5*np.array((( length+padding, width+padding),
                           (-length-padding, width+padding),
                           ( length+padding, -width-padding),
                           (-length-padding, -width-padding)))
    rotated_rel_corners = rotate_SO2(rel_corners, footprint[:,2])
    points = footprint[:,:2,None] + rotated_rel_corners.T
    AABB = np.stack((points.min(2),points.max(2)),axis=1)
    return AABB

def collisionAABBs(AABB, other_AABBs, spacing=0):
    """Detect collisions between axis-aligned bounding boxes

    :inputs:
        :AABB: axis aligned bounding box of target object. an AABB is np array
            of ((Ax,Ay),(Bx,By))
        :other_AABBs: list, AABB's of other objects to be checked against
        :spacing: extra space between bounding boxes. positive value means
        there must be space between them

    :return: boolean value indicating if collision exists
    """
    for other_AABB in other_AABBs:
        collision = AABB[0,0] < other_AABB[1,0]+spacing \
                and AABB[1,0] > other_AABB[0,0]-spacing \
                and AABB[0,1] < other_AABB[1,1]+spacing \
                and AABB[1,1] > other_AABB[0,1]-spacing
        if collision:
            return True
    return False

def containedInAABB(AABB, containerAABB, margin=0):
    """Checks that one AABB is fully contained in another AABB, this is useful
    for making sure a block is on top of platform.

    :inputs:
        :AABB: axis aligned bounding box of target object. an AABB is np array
            of ((Ax,Ay),(Bx,By))
        :containerAABB: AABB of another object that contains the first
        :margin: a positive margin means it must be within the container by at
        least some amount (e.g. it becomes a smaller container)

    :return: boolean value indicating if object is contained
    """
    return AABB[0,0] > containerAABB[0,0]+margin \
            and AABB[0,1] > containerAABB[0,1]+margin \
            and AABB[1,0] < containerAABB[1,0]-margin \
            and AABB[1,1] < containerAABB[1,1]-margin

def withinWorkspace(points, workspace, margin=0):
    """Checks if point is within 2d workspace given a margin

    :inputs:
        :points: xy coordinated as np array (N,2) or (2,)
        :workspace: 2x2 np array => ((X_low,X_high),(Y_low,Y_high))
        :margin: this acts like a padding, so a positive value means the point
        needs to lie within an area even smaller than the workspace

    :return: boolean array of size N
    """
    if len(points.shape) == 1:
        points = points[None,:]
    within = np.stack((points[:,0] >= workspace[0,0]+margin,
                       points[:,0] <= workspace[0,1]-margin,
                       points[:,1] >= workspace[1,0]+margin,
                       points[:,1] <= workspace[1,1]-margin), axis=1)
    if points.shape[0] == 1:
        return np.bitwise_and.reduce(within, axis=1)[0]
    return np.bitwise_and.reduce(within, axis=1)

def encode_state(structure, dist_flag=False):
    """Convert structure to encoding

    :structure: a string where each layer of the structure is represented
    by a 3 character string, with commas as delimiters.  The string reads from
    bottom of structure to top
    :dist_flag: boolean variable indicating if there are distractors in the
    structure
    :structured: boolean variable indicating if the encoding should be organized
    by layer identity or structure identity

    :return: np array of size (height,) and dtype=uint8 where height is the
    number of layers in the structure
    """
    convert = {'___' : 0,
               'c__' : 1,
               '__c' : 2,
               'c_c' : 3,
               '_b_' : 4,
               '_r_' : 5}
    return np.array([convert[layer] for layer in structure.split(',')]+[dist_flag],
                    dtype=np.uint8)

def decode_state(encoding):
    """Convert encoding to structure

    :encoding: np array of size (height,) and dtype=uint8 where height is the
    number of layers in the structure

    :return: a string where each layer of the structure is represented
    by a 3 character string, with commas as delimiters.  The string reads from
    bottom of structure to top
    """
    convert = ('___', 'c__', '__c', 'c_c', '_b_', '_r_')
    return ','.join([convert[i] for i in encoding[:-1]])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    footprint = np.array((0,0,0))
    length = 1
    width = 0.5
    AABB = getAABB(footprint, length, width)

    points = np.random.uniform(-1,1,size=(1000,2))
    # contained = np.array([containedInAABB(np.array((p,p)), AABB, margin=0.1) for p in points], dtype=bool)
    contained = np.array([collisionAABBs(np.array((p,p+0.01)), [AABB], padding=0.1) for p in points], dtype=bool)

    plt.figure()
    plt.plot(*AABB.T, 'b*')
    plt.plot(*points[contained].T, 'g.')
    plt.plot(*points[contained==False].T, 'r.')
    plt.show()
