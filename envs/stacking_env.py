import numpy as np
import numpy.random as npr
import os
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

import pybullet as pb
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from envs.utils import *

class StructureEncoder:
    def __init__(self,
                 max_num_layers,
                 platform_AABB,
                 block_size,
                 platform_height,
                ):
        self.max_height = max_num_layers
        self.platform_AABB = platform_AABB
        self.platform_footprint = np.array((*self.platform_AABB.mean(axis=0), 0.))
        self.layer_height = block_size
        self.platform_height = platform_height

        self.block_size = block_size
        self.distractor_padding = 0.25*block_size
        self.max_brick_offset = 0.5*block_size
        self.max_cube_offset = 0.25*block_size

        self.buildable_layers = ('c_c','_b_','y_y')

    def get_structure(self, objects, object_types):
        # filter objects 
        clutter_flag = self.filter_objects(objects, object_types)

        structure = []
        footprint = self.platform_footprint
        self.obj_under = None
        while True:
            layer, layer_clutter, footprint = self.label_layer(structure, footprint)
            clutter_flag = clutter_flag or layer_clutter

            structure.append(layer)
            if (layer not in self.buildable_layers) \
                    or len(structure) == self.max_height \
                    or layer_clutter:
                too_many_objs = (self.obj_heights >= len(structure)).any()
                clutter_flag = clutter_flag or too_many_objs
                break

        structure_desc = self.get_structure_desc(structure)
        return structure_desc, clutter_flag

    def label_layer(self, structure, footprint):
        layer_num = len(structure)
        idx_this_layer = np.where(self.obj_heights == layer_num)[0]

        if len(idx_this_layer) == 0:
            return '___', False, footprint

        elif len(idx_this_layer) == 1:
            this_type = self.obj_types[idx_this_layer[0]]
            if this_type in (constants.CUBE, constants.CYLINDER, constants.TRIANGLE):
                layer_desc = self.label_single_cube(idx_this_layer[0],
                                                    footprint,
                                                    this_type)
                if layer_num == 0 or self.valid_cube_placement(idx_this_layer):
                    self.obj_under = idx_this_layer
                    return layer_desc, False, footprint

            else: # BRICK or ROOF
                layer_desc = {constants.BRICK : '_b_',
                              constants.ROOF : '_r_'}[this_type]
                if layer_num == 0:
                    # overwrite footprint
                    footprint = self.get_footprint(idx_this_layer)
                if self.is_within(idx_this_layer[0], footprint,
                                            self.max_brick_offset):
                    self.obj_under = idx_this_layer
                    return layer_desc, False, footprint

        elif len(idx_this_layer) == 2:
            these_types = self.obj_types[idx_this_layer]
            if these_types[0] == these_types[1] \
                    and these_types[0] in (constants.CUBE, constants.TRIANGLE, constants.CYLINDER):
                obj_char = {constants.CUBE : 'c',
                            constants.TRIANGLE : 't',
                            constants.CYLINDER : 'y'}[these_types[0]]
                if layer_num == 0:
                    # make sure they are in right location
                    if self.valid_cube_placement(idx_this_layer):
                        # overwrite footprint
                        footprint = self.get_footprint(idx_this_layer)
                        self.obj_under = idx_this_layer
                        return f'{obj_char}_{obj_char}', False, footprint

                elif self.valid_cube_placement(idx_this_layer):
                    self.obj_under = idx_this_layer
                    return f'{obj_char}_{obj_char}', False, footprint

        # either too many objects or obj not placed well, so mark as clutter
        return '___', True, None

    def label_single_cube(self, obj_id, footprint, obj_type):
        pos = self.obj_positions[obj_id]
        footprint_vec = np.array((-np.sin(footprint[2]),
                                  np.cos(footprint[2])))
        sign = np.dot(footprint_vec, pos[:2] - footprint[:2])

        obj_char = {constants.CUBE : 'c',
                    constants.TRIANGLE : 't',
                    constants.CYLINDER : 'y'}[obj_type]
        return f'__{obj_char}' if sign > 0 else f'{obj_char}__'

    def get_structure_desc(self, structure=[]):
        n_layers_left = self.max_height - len(structure)
        structure.extend(n_layers_left * ['___'])

        return ','.join(structure)

    def get_footprint(self, obj_ids):
        pos = self.obj_positions[obj_ids]
        rot = self.obj_rotations[obj_ids]
        if len(obj_ids) == 1:
            return np.array((*pos[0,:2], rot[0,2]))
        else:
            vec = np.subtract(*pos[:,:2])
            return np.array((*pos[:,:2].mean(axis=0),
                             -np.arctan(vec[0]/vec[1])))

    def is_within(self, obj_id, footprint, max_offset):
        pos = self.obj_positions[obj_id]
        offset_dist = np.linalg.norm(pos[:2]-footprint[:2])

        return offset_dist < max_offset

    def valid_cube_placement(self, obj_ids):
        if self.obj_under is None:
            if len(obj_ids) == 1:
                obj_under_pos = self.platform_footprint
                obj_pos = self.obj_positions[obj_ids[0]]
                if np.linalg.norm(obj_pos[:2]-obj_under_pos[:2]) < 0.5*self.block_size:
                    return False
            elif not self.valid_cube_placement_base(obj_ids):
                return False
        elif len(self.obj_under) == 1:
            if len(obj_ids) == 1:
                obj_under_pos = self.obj_positions[self.obj_under[0]]
                obj_pos = self.obj_positions[obj_ids[0]]
                if np.linalg.norm(obj_pos[:2]-obj_under_pos[:2]) < 0.5*self.block_size:
                    return False
            elif len(obj_ids) == 2:
                if not self.valid_cube_placement_base(obj_ids):
                    return False
        elif len(self.obj_under) == 2:
            obj_under_pos = self.obj_positions[self.obj_under]
            for obj_id in obj_ids:
                if not (self.is_within(obj_id, obj_under_pos[0], self.max_cube_offset) \
                       or self.is_within(obj_id, obj_under_pos[1], self.max_cube_offset)):
                    return False
        else:
            raise TypeError
        return True

    def valid_cube_placement_base(self, obj_ids):
        if len(obj_ids) == 1:
            return True
        obj_pos = np.array([self.obj_positions[i] for i in obj_ids])
        vec = (obj_pos[1]-obj_pos[0])[:2]
        dist = np.linalg.norm(vec)
        angle = -np.arctan(vec[0]/vec[1])
        if 1.1*self.block_size < dist < 2.25*self.block_size \
               and np.abs(angle) < 0.4:
            return True
        return False

    def filter_objects(self, all_objects, all_object_types):
        '''Get all objects that exist on platform'''
        self.obj_handles = []
        self.obj_types = []
        self.obj_positions = []
        self.obj_rotations = []

        clutter_flag = False
        for o in all_objects:
            AABB = np.array(o.getBoundingBox())[:,:2]
            if containedInAABB(AABB, self.platform_AABB):
                self.obj_handles.append(o)
                self.obj_types.append(all_object_types[o])
                pos, quat = o.getPose()
                self.obj_positions.append(pos)
                self.obj_rotations.append(pb.getEulerFromQuaternion(quat))
            elif collisionAABBs(AABB, [self.platform_AABB],
                            spacing=-self.distractor_padding):
                clutter_flag = True

        self.obj_types = np.array(self.obj_types)
        self.obj_positions = np.array(self.obj_positions)
        self.obj_rotations = np.array(self.obj_rotations)

        self.obj_heights = np.array([])
        if self.obj_positions.size > 0:
            self.assign_heights(self.obj_positions[:,2])
        return clutter_flag

    def assign_heights(self, z_pos):
        """Given heights of objects on platform, assign a integer layer to
        each, where an object on ground is given layer 0

        :inputs:
            :z_pos: z coordinates for objects as np.array (N,)
        :returns:
            :heights_error: bool indicating unexpected height value
        """
        heights = (z_pos - 0.5*self.layer_height \
                  - self.platform_height)/self.layer_height
        self.obj_heights = np.round(heights).astype(int)

class StackingEnv(PyBulletEnv):
    def __init__(self, config):
        self.obs_size = config.get('obs_size', 90)
        self.side_view_size = config.get('side_view_size', 90)

        self.max_height = config.get('max_height', 4)
        self.distractor_max_angle = config.get('distractor_max_angle', 0.01)

        self.side_view_distance = config.get('side_view_distance', 2)
        self.side_view_yaw = config.get('side_view_yaw', 90)
        self.side_view_pitch = config.get('side_view_pitch', -50)
        self.side_view_fov = config.get('side_view_fov', 6)

        # default params
        config['robot'] = config.get('robot', 'kuka')
        config['workspace'] = np.array(((0.35,0.65),
                                        (-0.15,0.15),
                                        (0,1)))
        config['max_steps'] = 100
        config['simulate_grasp'] = True
        config['fast_mode'] = True
        config['random_orientation'] = True

        self.platform_dim = np.array((0.08,0.12,0.01))
        self.platform_offset = 0.015
        self.platform_ws = np.array((
            (config['workspace'][0,1]-self.platform_offset-self.platform_dim[0],
             config['workspace'][0,1]-self.platform_offset),
            (config['workspace'][1,0]+self.platform_offset,
             config['workspace'][1,0]+self.platform_offset+self.platform_dim[1])
                                    ))

        self.platform_AABB = self.platform_ws.T
        self.workspace_margin = 0.015
        self.platform_margin = 0.005
        self.object_spacing = 0.02
        self.block_scale = 0.6
        self.brick_width = 0.05
        self.brick_length = 0.15
        self.brick_height = 0.05

        npr.seed(config['seed'])

        obj_height = self.block_scale*self.brick_height
        self.in_hand_range = (0.0, obj_height)
        self.top_down_range = (0.0,
                               self.platform_dim[2] + self.max_height*obj_height)
        self.side_view_range = (0.69, 0.84)

        super(StackingEnv, self).__init__(config)

        # restrict gripper opening to improve performnace in clutter
        if config['robot'] == 'kuka':
            self.robot.gripper_joint_limit[1] = config.get('max_finger_angle', 0.12)
        elif config['robot'] == 'ur5':
            self.robot.gripper_joint_limit[0] = config.get('max_finger_angle', 0.01)

        self.initialize()
        self.soft_reset_counter = 0
        self.hard_reset_freq = 100

        self.encoder = StructureEncoder(self.max_height,
                                        self.platform_AABB,
                                        self.block_scale*self.brick_width,
                                        self.platform_dim[2])

    def reset(self, suff_objs):
        """Resets objects in pybullet simulator for to a start structure such
        that the goal structure is achievable (e.g. sufficient blocks)

        :inputs:
            :suff_objs: dictionary of describing how many of each object must
            exist in the scene

        :return:
            :obs: observation tuple, see self._getObservation
        """
        self.soft_reset_counter += 1
        if self.soft_reset_counter % self.hard_reset_freq == 0:
            self.initialize()
            self.soft_reset_counter = 0

        while True:
            self._resetSim()
            while True:
                # sometimes placing istractors can be difficult so we
                # allow multiple attempts 
                try:
                    self._placeDistractors(suff_objs)
                    self.wait(100)
                except NoValidPositionException:
                    pass
                else:
                    break
            break
        return self._getObservation()

    def step(self, action):
        """Step environment by taking an action

        :inputs:
            :action: array of 3 ints (prim, px, py)

        :return:
            :obs: tuple (is_holding, in_hand, top_down, side_view), see
                self._getObservation() for more info
            :done: Boolean flag indicating if any objects have been knocked out
                off workspace
        """
        prim, px, py = action
        x,y = self._getPosFromPixels(px, py)
        action = (prim, x, y)

        self.takeAction(action)
        self.wait(100)
        obs = self._getObservation(action)

        done = not self._allObjectsInWorkspace()

        return obs, done

    def random_action(self):
        return np.array((np.random.randint(2),
                         np.random.randint(0, self.obs_size),
                         np.random.randint(0, self.obs_size)), dtype=np.uint8)

    def _allObjectsInWorkspace(self):
        """Checks if all objects are in workspace, excluding objects in hand

        :return: boolean flag that is true if all objects are in workspace,
        false otherwise
        """
        for o in self.objects:
            if self._isObjectHeld(o):
                continue
            AABB = np.array(o.getBoundingBox())[:,:2]
            if not withinWorkspace(AABB, self.workspace).all():
                return False
        return True

    def _getPosFromPixels(self, px, py):
        """Converts pixel positions of top_down image to positions in the
        simulator, where the image is indexed like I[px,py].

        :inputs:
            :px: index of image in first dimension
            :py: index of image in second dimension

        :returns:
            :x: x-coordinate in simulator
            :y: y-coordinate in simulator
        """
        x = px * self.heightmap_resolution + self.workspace[0][0]
        y = py * self.heightmap_resolution + self.workspace[1][0]

        return x, y

    def _takeSnapshot(self, depth=True):
        """Returns image of the scene according to specified camera parameters
        """
        cam_target_pos = np.array([self.platform_ws[0].mean()+0.12, self.platform_ws[1].mean(), 0.06])
        # view_matrix = pb.computeViewMatrixFromYawPitchRoll(cam_target_pos,
                                                           # distance=self.side_view_distance,
                                                           # yaw=self.side_view_yaw,
                                                           # pitch=self.side_view_pitch,
                                                           # roll=0,
                                                           # upAxisIndex=2)
        cam_vec = np.array((-0.45,0,0.10))
        view_matrix = pb.computeViewMatrix(cam_target_pos + cam_vec,
                                           cam_target_pos,
                                           [0,0,1])

        proj_matrix = pb.computeProjectionMatrixFOV(fov=40,
                                                    aspect=1,
                                                    nearVal=0.14,
                                                    farVal=0.7)

        image_arr = pb.getCameraImage(width=self.side_view_size,
                                      height=self.side_view_size,
                                      viewMatrix=view_matrix,
                                      projectionMatrix=proj_matrix)
        if depth:
            return image_arr[3]
        return image_arr[2]

    def _getObservation(self, action=None):
        """Returns observation of current simulator state

        :input:
            :action: np array (prim, px, py)

        :return:
            :is_holding: Boolean value indicating if anything is in gripper
            :in_hand_img: top down image of scene where grasp took place
            :top_down_img: top down image of workspace
            :side_view_img: image of scene from alternate angle
        """
        old_heightmap = self.heightmap
        self.heightmap = self._getHeightmap()
        top_down_img = self.heightmap.copy()[:,:,None]

        if action is None or self._isHolding() == False:
            in_hand_img = self.getEmptyInHand()
        else:
            motion_primative, x, y, z, rot = self._decodeAction(action)
            in_hand_img = self.getInHandImage(old_heightmap, x, y, z,
                                              rot, self.heightmap)
            bg_val = max(in_hand_img.max()-self.block_scale*self.brick_height,0)
            in_hand_img = np.clip(in_hand_img - bg_val, 0, 1)
        side_view = self._takeSnapshot()[:,:,None]

        obs = (self._isHolding(),
                in_hand_img[0][...,None],
                top_down_img,
                side_view,
               )
        return self._normalizeObservation(obs)

    def _normalizeObservation(self, obs):
        """Convert images in observation generated by pybullet from floats
        to np.uint8.  This makes them more compact for storage/memory buffer

        :obs: see self._getObservation for details
        """
        is_holding, in_hand, top_down, side_view = obs
        in_hand = normalize_img(in_hand, self.in_hand_range)
        top_down = normalize_img(top_down, self.top_down_range)
        side_view = normalize_img(side_view, self.side_view_range)
        return is_holding, in_hand, top_down, side_view

    def initialize(self):
        """Initializes pybullet sim, and loads necessary URDFs
        """
        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSubSteps=0,
                                 numSolverIterations=50,
                                 solverResidualThreshold=1e-7,
                                 constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
        pb.setTimeStep(self._timestep)
        pb.setGravity(0, 0, -10)

        # offset grid to make it one-color background
        self.table_id = pb.loadURDF('plane.urdf', [.1,0.5,0])
        pb.changeDynamics(self.table_id, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        #load platform
        urdf_path = os.path.join(os.getcwd(),
                                 f'{os.getcwd()}/envs/platform.urdf')
        platform_pos = np.array((*self.platform_ws.mean(axis=1),
                                 0.5*self.platform_dim[2]))
        self.platform_id = pb.loadURDF(urdf_path, platform_pos)

        # Load the UR5 and set it to the home positions
        self.robot.initialize()

        # Reset episode vars
        self.objects = list()
        self.object_types = {}

        self.heightmap = None
        self.current_episode_steps = 1
        self.last_action = None

        # Step simulation
        pb.stepSimulation()

    def _resetSim(self):
        """Reset sim state by clearing all objects, and memory variables
        """
        for o in self.objects:
            pb.removeBody(o.object_id)
        self.robot.reset()
        self.objects = list()
        self.object_types = {}
        self.heightmap = None
        self.last_action = None
        self.last_obj = None
        self.state = {}
        self.pb_state = None
        pb.stepSimulation()

    def _sample_nonplatform(self, length, width, count):
        poses = np.zeros((count, 3))
        poses[:,2] = npr.uniform(-self.distractor_max_angle,
                                 self.distractor_max_angle, size=count)
        n_valid = 0
        offsets = 0.5*np.array((length, width))+self.workspace_margin
        low = self.workspace[:2,0]+offsets
        extent = self.workspace[:2,1]-self.workspace[:2,0]-2*offsets
        while n_valid != count:
            xy = npr.uniform(size=(count*5,2))*extent + low
            valid_mask = np.bitwise_not(withinWorkspace(xy, self.platform_ws))
            n_to_add = min(len(poses)-n_valid, valid_mask.sum())
            poses[n_valid:n_valid+n_to_add,:2] = xy[valid_mask][:n_to_add]
            n_valid += n_to_add
        return poses

    def getState(self):
        """Determines structure present on platform and whether distractors are
        present.

        :returns:
            :structure: string where each layer is 3char separated by comma
            :distractors_present: boolean value
        """
        return self.encoder.get_structure(self.objects, self.object_types)

    def _addObject(self, o_char, position, angle, scale):
        """Add object to simulator, and append to self.objects,
        self.object_types

        :inputs:
            :o_char: character in ('c','b','r','t') describing object
            :position: np array (x,y,z)
            :angle: rotation angle around z-axis
            :scale: block scale value
        :returns:
            :handle: object handle
        """
        orientation = pb.getQuaternionFromEuler([0., 0., angle])
        handle = {'c' : pb_obj_generation.generateCube,
                  'b' : pb_obj_generation.generateBrick,
                  'r' : pb_obj_generation.generateRoof,
                  't' : pb_obj_generation.generateTriangle,
                  'y' : pb_obj_generation.generateCylinder,
                 }[o_char](position, orientation, scale)

        self.objects.append(handle)
        self.object_types[handle] = {
            'c' : constants.CUBE,
            'b' : constants.BRICK,
            'r' : constants.ROOF,
            't' : constants.TRIANGLE,
            'y' : constants.CYLINDER,
        }[o_char]
        return handle

    def _placeDistractors(self, suff_objs):
        """Place distractors in the scene such that all objects exist in the
        scene.

        :inputs:
            :suff_objs: dict that stores count for each object type
        :returns:
            :objects: list of tuples (obj_handle, xyz_position)
        """
        def still_valid_mask(collision_AABB, proposal_AABBs,
                             spacing=self.object_spacing):
            # check that none of the corners in proposal AABBs is in existing AABBs
            proposal_points = proposal_AABBs.reshape(-1,2)

            # transpose of AABB is workspace
            # margin is the opposite of spacing in this context
            points_collision_mask = withinWorkspace(proposal_points,
                                                    collision_AABB.T,
                                                    margin=-spacing)
            # neither of the 2 points can be in collision
            AABB_collision_mask = np.bitwise_or.reduce(
                                    points_collision_mask.reshape(-1,2),
                                    axis=1
            )
            return np.bitwise_not(AABB_collision_mask)

        def rec_add_objects(existing_AABBs,
                            proposal_AABBs,
                            proposal_footprints,
                            n_to_add):
            n_attempts = 30
            if n_to_add == 0:
                return []
            if len(proposal_AABBs) < n_attempts:
                return []
            for proposal_id in np.random.randint(0, len(proposal_AABBs), size=n_attempts):
                new_AABB = proposal_AABBs[proposal_id]
                if not collisionAABBs(new_AABB, existing_AABBs, self.object_spacing):
                    existing_AABBs.append(new_AABB)
                    mask = still_valid_mask(new_AABB, proposal_AABBs)
                    return [proposal_footprints[proposal_id]] + rec_add_objects(
                                                       existing_AABBs,
                                                       proposal_AABBs[mask],
                                                       proposal_footprints[mask],
                                                       n_to_add-1)
            return []

        brick_length = self.brick_width*self.block_scale
        brick_width = self.brick_length*self.block_scale
        cube_size = self.brick_width*self.block_scale
        brick_footprints = self._sample_nonplatform(brick_length, brick_width, 1000)
        brick_AABBs = vec_getAABB(brick_footprints, brick_length, brick_width)
        cube_footprints = self._sample_nonplatform(cube_size, cube_size, 2000)
        cube_AABBs = vec_getAABB(cube_footprints, cube_size, cube_size)

        existing_AABBs = [self.platform_AABB]

        obj_args = []
        # we want to try the larger blocks first
        for obj_char in 'brcty':
            if obj_char == 'c':
                valid_cube_mask = still_valid_mask(existing_AABBs[0],
                                                   cube_AABBs,
                                                   spacing=0)
                for AABB in existing_AABBs[1:]:
                    valid_cube_mask = np.bitwise_and(valid_cube_mask,
                                                     still_valid_mask(AABB,
                                                                      cube_AABBs,
                                                                      spacing=self.object_spacing)
                                                    )
                cube_footprints = cube_footprints[valid_cube_mask]
                cube_AABBs = cube_AABBs[valid_cube_mask]

            count = suff_objs.get(obj_char, 0)
            if count == 0:
                continue
            is_wide = obj_char in 'br'
            AABBs = brick_AABBs if is_wide else cube_AABBs
            footprints = brick_footprints if is_wide else cube_footprints

            success = False
            for _ in range(50):
                tmp_existing_AABBs = list(existing_AABBs)
                tmp_proposal_AABBs = AABBs.copy()
                tmp_proposal_footprints = footprints.copy()
                new_footprints = rec_add_objects(tmp_existing_AABBs,
                                                 tmp_proposal_AABBs,
                                                 tmp_proposal_footprints,
                                                 count)
                if len(new_footprints) == count:
                    success = True
                    existing_AABBs = tmp_existing_AABBs
                    for footprint in new_footprints:
                        position = np.array((*footprint[:2],
                                             0.5*cube_size))
                        obj_args.append((obj_char, position,
                                         footprint[2], self.block_scale))
                    if is_wide:
                        brick_AABBs = tmp_proposal_AABBs
                        brick_footprints = tmp_proposal_footprints
                    else:
                        cube_AABBs = tmp_proposal_AABBs
                        cube_footprints = tmp_proposal_footprints
                    break
            if success == False:
                raise NoValidPositionException

        objects = []
        for arg in obj_args:
            handle = self._addObject(*arg)
            objects.append((handle, arg[1]))
        return objects

def generate_structure_pics(fdir, other_obj_chars=[], max_height=5):
    from configs import get_env_configs, get_training_configs
    from envs.goal_space import StructuredGoalSpace
    import matplotlib.pyplot as plt
    import os
    from PIL import Image

    if not os.path.exists(fdir):
        os.mkdir(fdir)

    config = get_env_configs(max_height=max_height,
                             other_obj_chars=other_obj_chars,
                             side_view_pitch=-40,
                             side_view_fov=20,
                             side_view_size=320)
    env = StackingEnv(config)

    base_position = np.zeros(3)
    base_position[:2] = env.platform_ws[:, :2].mean(axis=1)
    base_position[2] = env.platform_dim[2] + 0.5 * env.brick_height * env.block_scale
    cube_offset = np.array((0,0.8*env.block_scale*env.brick_width,0))

    space = StructuredGoalSpace(env.max_height)
    if 't' in other_obj_chars:
        space.add_triangle_object()
    if 'y' in other_obj_chars:
        space.add_cylinder_object()

    files = []
    for terminal_goal in space.terminal_goals:
        env._resetSim()
        pos =  np.copy(base_position)
        for layer in terminal_goal.split(','):
            if layer == '___':
                break
            if layer[0] != '_':
                env._addObject(layer[0], pos-cube_offset, 0, env.block_scale)
            if layer[2] != '_':
                env._addObject(layer[2], pos+cube_offset, 0, env.block_scale)
            if layer[1] != '_':
                env._addObject(layer[1], pos, 0, env.block_scale)
            pos[2] += env.brick_height*env.block_scale

        # change appearance
        for o in env.objects:
            if env.object_types[o] == constants.TRIANGLE:
                pb.changeVisualShape(o.object_id, -1, rgbaColor=[117/255, 31/255, 202/255,1])
            elif env.object_types[o] == constants.CYLINDER:
                pb.changeVisualShape(o.object_id, -1, rgbaColor=[242/255, 234/255, 5/255,1])

        obs = env._takeSnapshot(depth=False)
        im = Image.fromarray(obs)
        fname = os.path.join(fdir,f'{terminal_goal}.png')
        im.save(fname)

if __name__ == "__main__":
    from configs import get_env_configs, get_training_configs
    import time
    config = get_env_configs(render=1, robot='ur5', max_height=5,
                             side_view_fov=50,
                             side_view_pitch=-80,
                             side_view_size=60)
    env = StackingEnv(config)
    env.reset({'c' : 2,'b':1,'r' : 1})
    corners = env.workspace.copy()
    corners = ((0.35,-0.15,0.002),
               (0.65, -0.15,0.002),
               (0.65, 0.15,0.002),
               (0.35, 0.15,0.002))
    for i in range(4):
        pb.addUserDebugLine(corners[i],
                            corners[(i+1)% 4],
                            lineColorRGB=(0,0,0),
                            lineWidth=3)

    # add the blocks
    # position = np.zeros(3)
    # position[:2] = env.platform_ws[:, :2].mean(axis=1)
    # position[2] = env.platform_dim[2] + 0.5 * env.brick_height * env.block_scale

    # env._addObject('c', position-np.array((0,0.025,0)), 0, env.block_scale)
    # env._addObject('c', position+np.array((0,0.025,0)), 0, env.block_scale)
    # position[2] += env.brick_height * env.block_scale
    # env._addObject('r', position, 0, env.block_scale)
    # position[2] += env.brick_height * env.block_scale
    # env._addObject('c', position-np.array((0,0.025,0)), 0, env.block_scale)
    # env._addObject('c', position+np.array((0,0.025,0)), 0, env.block_scale)
    # position[2] += env.brick_height * env.block_scale
    # env._addObject('r', position, 0, env.block_scale)
    # # # for i in range(env.max_height-1):
        # # # env._addObject('b', position, 0, env.block_scale)
        # # # position[2] += env.brick_height * env.block_scale
    # # # env._addObject('b', position, 0, env.block_scale)

    # suff_objs = {'c':4,'b':2,'r':1}
    # for _ in range(1):
        # # env.initialize()
        # t = time.time()
        # try:
            # env._placeDistractors(suff_objs)
            # print((time.time()-t)*1000)
        # except NoValidPositionException:
            # pass
        # env.wait(100)

    while 1:
        time.sleep(0.1)

    sv = env._takeSnapshot()
    # _, ih, td, sv = env._getObservation()
    # print(ih.shape)
    # print(td.shape)
    # print(sv.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sv)
    # plt.axis('off')
    plt.show()
    exit()
