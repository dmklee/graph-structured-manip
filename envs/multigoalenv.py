import numpy as np
import numpy.random as npr
from envs.stacking_env import StackingEnv
from envs.goal_space import StructuredGoalSpace
from envs.utils import encode_state, decode_state

class MultiGoalEnv(StackingEnv):
    def __init__(self,
                 config,
                 specific_goal=None
                ):
        super().__init__(config)
        self.goal_space = StructuredGoalSpace(self.max_height, config['seed'])
        if specific_goal is not None:
            self.goal_space.constrain_to(specific_goal)

        for obj_char in config['other_obj_chars']:
            if obj_char == 't0':
                self.goal_space.add_triangle_object(id_=0)
            elif obj_char == 't1':
                self.goal_space.add_triangle_object(id_=1)
            elif obj_char == 'y':
                self.goal_space.add_cylinder_object()

        self.include_all_objects = config.get('include_all_objects', True)
        self.num_distractors = config.get('num_distractors', 0)

        # should state/goal encodings be organized by layer or represented as one hot
        self.use_structured_encodings = config.get('use_structured_encodings',
                                                  True)

    def reset(self, goal_distribution='all', goal_subset=None):
        """Overwrites SymbolicStructureEnv method by sampling the state and
        goal structures with the goal_space.
        """
        assert goal_distribution in ('all','terminal')

        if goal_subset is not None:
            idx = np.random.randint(len(goal_subset))
            self.goal = goal_subset[idx]
        elif goal_distribution == 'all':
            self.goal = self.goal_space.sample()
        elif goal_distribution == 'terminal':
            self.goal = self.goal_space.sample_terminal()
        else:
            raise TypeError

        self.opt_path_length = self.goal_space.opt_dist(self.goal)

        if self.include_all_objects:
            suff_objs = self.goal_space.sufficient_objects()
        else:
            # only include objs needed to achieve goal
            suff_objs = self.goal_space.sufficient_objects(self.goal)

        for _ in range(self.num_distractors):
            obj_char = self.goal_space.obj_chars[npr.randint(len(self.goal_space.obj_chars))]
            suff_objs[obj_char] += 1

        obs = super(MultiGoalEnv, self).reset(suff_objs)
        s_enc = self._encode(self.goal_space.root)
        g_enc = self._encode(self.goal)
        return obs, s_enc, g_enc

    def _encode(self, struct, dist_flag=False):
        if self.use_structured_encodings:
            return encode_state(struct, dist_flag)
        else:
            # repesent encoding as onehot over all structures + one bit for distractor
            try:
                struct_id = self.goal_space.structure_dict[struct]
            except KeyError:
                # struct is not in space so label distractor
                struct_id = 0
                dist_flag = True
            return np.array((struct_id, dist_flag), dtype=np.uint8)

    def _decode(self, enc):
        if self.use_structured_encodings:
            return decode_state(enc)
        else:
            # get structure index from onehot
            struct_id = enc[0]
            return self.goal_space.all_structures[struct_id]

    def step(self, action):
        """Overwrites SymbolicStructureEnv method to include structure state"""
        obs, done = super(MultiGoalEnv, self).step(action)
        state, dist_flag = self.getState()
        s_enc = self._encode(state, dist_flag)
        return obs, s_enc, done

    def get_subgoal(self, s_enc):
        """Uses goal_space to return subgoal on path to self.goal
        """
        if s_enc[-1]:
            # if distractor, point toward same state no distractor
            subgoal_enc = s_enc.copy()
            subgoal_enc[-1] = 0
            return subgoal_enc
        state = self._decode(s_enc)
        if state == self.goal:
            subgoal_enc = s_enc.copy()
            subgoal_enc[-1] = 0
            return subgoal_enc
        subgoal = self.goal_space.get_subgoal(state, self.goal)
        return self._encode(subgoal, False)

    def enc_to_desc(self, enc):
        dist = '?' if enc[-1] else ' '
        if self.use_structured_encodings:
            layer = ('___','c__','__c','c_c','_b_','_r_')
            state = enc[:-1]
            return ','.join([layer[e] for e in state]+[dist])
        else:
            structure = self.goal_space.all_structures[enc[0]]
            return ','.join([structure, dist])
        # return '\n'.join([dist] + [layer[e] for e in state[::-1]])

def createMultiGoalEnv(ignore, config, specific_goal=None):
    def _thunk():
        return MultiGoalEnv(config, specific_goal)

    return _thunk
