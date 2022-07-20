import numpy as np

class GoalSpace:
    def __init__(self, max_height, goal_subset=None):
        self.max_height = max_height
        self.layers = ('___','c__','__c','c_c','_b_','_r_')

        self.root =  ','.join(['___' for _ in range(max_height)])

        self.in_layer_transitions = {
            '___' : ('c__','__c','_b_','_r_'),
            'c__' : ('c_c',),
            '__c' : ('c_c',),
            '_r_' : (),
            '_b_' : (),
            'c_c' : (),
        }
        self.buildable_layers = ('_b_','c_c')
        self.addon_layers = ('c__','__c','_b_','_r_')

        self._build_structures(self.root)

        if goal_subset is not None:
            assert all([g in self.all_structures for g in goal_subset])
            self.goals = tuple(goal_subset)
            self.terminal_goals = tuple([g for g in self.goals if self.is_terminal(g)])

        self.all_sufficient_objects = {k:0 for k in 'cbr'}
        for struct in self.all_structures:
            for k in 'cbr':
                self.all_sufficient_objects[k] = max(self.all_sufficient_objects[k],
                                               struct.count(k))

    def _build_structures(self, root):
        all_structures = set()
        all_paths = {root : []}

        queue = [root]

        while len(queue) > 0:
            structure = queue.pop()
            if structure in all_structures:
                continue
            else:
                all_structures.add(structure)

            nbrs = self._build_step(structure)
            for nbr in nbrs:
                queue.append(nbr)
                all_paths[nbr] = all_paths[structure].copy() + [structure]

        self.paths = {}
        # for consistency, all paths will place left block first
        for goal, path in all_paths.items():
            path = [p.replace('__c','c__').replace('__t','t__').replace('__y','y__')
                                      for p in path]
            self.paths[goal] = tuple(path)

        self.all_structures = tuple(sorted(list(all_structures)))

        all_structures.remove(root)
        # goals are structures (not including base) and without redundant single
        # cube layers e.g no __c
        self.goals = tuple([s for s in all_structures if s.count('__c')==0])
        self.terminal_goals = tuple([g for g in self.goals if self.is_terminal(g)])

    def _build_step(self, structure):
        height = self.get_structure_height(structure)
        if height == 0:
            layer = '___'
        else:
            layer = structure[4*height-4:4*height-1]

        next_structures = []

        # perform in layer changes
        for new_layer in self.in_layer_transitions[layer]:
            if height == 0:
                new_structure = new_layer + structure[3:]
            else:
                below = structure[:4*max(height-1,0)]
                new_structure = below + new_layer + structure[4*height-1:]
            next_structures.append(new_structure)

        # perform addition of new layers
        if 0 < height < self.max_height and layer in self.buildable_layers:
            for addon in self.addon_layers:
                new_structure = structure[:4*height] + addon + structure[4*height+3:]
                next_structures.append(new_structure)

        return next_structures

    def is_terminal(self, structure):
        if self.get_structure_height(structure) == self.max_height:
            return structure[-3:] in ('c_c','_r_','_b_','t_t','y_y')
        else:
            # must have roof
            return ('r' in structure or structure.count('t') == 2)

    def get_structure_height(self, structure):
        return self.max_height - structure.count('___')

    def __len__(self):
        return len(self.goals)

class StructuredGoalSpace(GoalSpace):
    def __init__(self, max_height, seed=None):
        super().__init__(max_height)
        self.novel_structures = set()
        self.obj_chars = 'cbr'

        if seed is not None:
            np.random.seed(seed)

        self.structure_dict = {k:i for i,k in enumerate(self.all_structures)}

    @property
    def n_goals(self):
        return len(self)

    def opt_dist(self, goal):
        return len(self.paths[goal])

    def get_subgoal(self, start, goal):
        ''' Start and goal are structure strings'''
        if start == goal:
            return goal
        path = self.get_path(start, goal)
        try:
            current_idx = path.index(start)
        except ValueError:
            # print(start, goal, path)
            return goal
        return path[current_idx+1]

    def get_extended_subgoal(self, start, goal):
        ''' Start and goal are structure strings'''
        if start == goal:
            return goal
        path = self.get_path(start, goal)
        try:
            current_idx = path.index(start)
        except ValueError:
            return goal
        subgoal = path[np.random.randint(current_idx, len(path))]
        return subgoal

    def get_path(self, start, goal):
        ''' Start and goal are structure strings'''
        path = list(self.paths[goal]) + [goal]
        return path

    def sample(self):
        return self.goals[np.random.randint(self.n_goals)]

    def sample_terminal(self):
        term_goals = self.terminal_goals
        return term_goals[np.random.randint(len(term_goals))]

    def sufficient_objects(self, goal=None):
        if goal is None:
            return self.all_sufficient_objects
        else:
            return {k: goal.count(k) for k in self.obj_chars}

    def constrain_to(self, goal):
        goals = list(self.paths[goal][1:]) + [goal]
        self.goals = tuple(goals)
        self.terminal_goals = tuple([goal])

        self.all_structures = tuple([self.root] + goals.copy())
        self.structure_dict = {k:i for i,k in enumerate(self.all_structures)}

        self.paths = {g : self.paths[g] for g in self.goals}

    def add_triangle_object(self, id_=0):
        self.layers = tuple(list(self.layers) + ['t__', '__t', 't_t'])

        self.in_layer_transitions['___'] = ('c__','__c','_b_','_r_','t__','__t')
        self.in_layer_transitions['t__'] = ('t_t',)
        self.in_layer_transitions['__t'] = ('t_t',)
        self.in_layer_transitions['t_t'] = ()

        self.addon_layers = tuple(list(self.addon_layers) + ['t__','__t'])

        # we need to preserve the ordering of the first goal space
        old_all_structures = tuple(self.all_structures)

        # rebuild all structures so we get the paths
        self._build_structures(self.root)
        new_structure = ['_b_,_b_,c_c,_b_,t_t',
                         'c_c,_b_,_b_,c_c,t_t'][id_]

        # ensure the ordering is the same
        all_structures = list(old_all_structures)
        for other in self.get_path(self.root, new_structure):
            if other not in all_structures:
                all_structures.append(other)
                self.novel_structures.add(other)

        self.all_structures = tuple(all_structures)

        self.structure_dict = {k:i for i,k in enumerate(self.all_structures)}

        self.goals = tuple([s for s in self.all_structures if s.count('t')])
        self.terminal_goals = tuple([s for s in self.all_structures if s.count('t_t')])

        self.obj_chars = self.obj_chars + 't'

    def add_cylinder_object(self):
        self.layers = tuple(list(self.layers) + ['y__', '__y', 'y_y'])

        self.in_layer_transitions['___'] = ('c__','__c','_b_','_r_','y__','__y')
        self.in_layer_transitions['y__'] = ('y_y',)
        self.in_layer_transitions['__y'] = ('y_y',)
        self.in_layer_transitions['y_y'] = ()
        self.buildable_layers = tuple(list(self.buildable_layers) + ['y_y'])

        self.addon_layers = tuple(list(self.addon_layers) + ['y__','__y'])

        # we need to preserve the ordering of the first goal space
        old_all_structures = tuple(self.all_structures)

        # rebuild all structures so we get the paths
        self._build_structures(self.root)
        new_structure = 'c_c,_b_,y_y,_b_,_r_'
        # all_structures = list(self.all_structures)
        # all_structures.extend(['_b_,c_c,y__,___,___',
                               # '_b_,c_c,__y,___,___',
                               # '_b_,c_c,y_y,___,___',
                               # '_b_,c_c,y_y,_b_,___',

        # ensure the ordering is the same
        all_structures = list(old_all_structures)
        for other in self.get_path(self.root, new_structure):
            if other not in all_structures:
                all_structures.append(other)
                self.novel_structures.add(other)
        self.all_structures = tuple(all_structures)

        self.structure_dict = {k:i for i,k in enumerate(self.all_structures)}

        self.goals = tuple([s for s in self.all_structures if s.count('y_y')])
        self.terminal_goals = tuple([g for g in self.goals if self.is_terminal(g)])

        self.obj_chars = self.obj_chars + 'y'

def plot_structures(structures):

    def add_structure(position, structure):
        unit = 1/(structure.count(',')+2)
        x,y = position
        patches = []
        colors = sns.color_palette('colorblind', n_colors=3)
        brick_color = 'w'# colors[0]
        cube_color = 'r'# colors[1]
        roof_color = 'k'# colors[2]
        for l_id, layer in enumerate(structure.split(',')):
            if layer == '_b_':
                new_patch = mpatch.Rectangle((x,l_id*unit+y),3*unit,unit,
                                             fc=brick_color, ec='k')
                patches.append(new_patch)
            elif layer == '_r_':
                vertices = np.array(((x,l_id*unit+y),
                                     (x+1.5*unit,(l_id+1)*unit+y),
                                     (x+3*unit,l_id*unit+y)))
                new_patch = mpatch.Polygon(vertices, fc=roof_color, ec='k')
                patches.append(new_patch)
            elif layer == 'c__':
                new_patch = mpatch.Rectangle((x+0.25*unit,l_id*unit+y),unit,unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
            elif layer == '__c':
                new_patch = mpatch.Rectangle((x+1.75*unit,l_id*unit+y),unit,unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
            elif layer == 'c_c':
                new_patch = mpatch.Rectangle((x+0.25*unit,l_id*unit+y),unit,unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
                new_patch = mpatch.Rectangle((x+1.75*unit,l_id*unit+y),unit,unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
        return patches

    n_structures = len(structures)
    width = int(n_structures**0.5+1)

    # structures = sorted(structures)
    # structures = sorted(structures, key=lambda x: x.count('___'), reverse=False)

    width = 12
    plt.figure()
    ax = plt.gca()
    for i in range(width):
        for j in range(n_structures//width + 1):
            try:
                struct = structures[i + j*width]
                position = np.array((i*0.7, j))
                [ax.add_patch(p) for p in add_structure(position, struct)]
            except IndexError:
                break

    plt.axis('off')
    # plt.axis('equal')
    plt.xlim((-0.1, width*0.7))
    plt.ylim((-0.1, n_structures//width+1))
    plt.tight_layout()
    # plt.savefig('height5-structures.png')
    # plt.close()
    # plt.show()

def show_all():
    def add_structure(position, structure):
        h_unit = h/6
        w_unit = w/4
        x,y = position
        patches = []
        colors = sns.color_palette('colorblind', n_colors=3)
        brick_color = 'w'# colors[0]
        cube_color = 'r'# colors[1]
        roof_color = 'k'# colors[2]
        for l_id, layer in enumerate(structure.split(',')):
            if layer == '_b_':
                new_patch = mpatch.Rectangle((x,l_id*h_unit+y),3*w_unit,h_unit,
                                             fc=brick_color, ec='k')
                patches.append(new_patch)
            if layer == '_r_':
                vertices = np.array(((x,l_id*h_unit+y),
                                     (x+1.5*w_unit,(l_id+1)*h_unit+y),
                                     (x+3*w_unit,l_id*h_unit+y)))
                new_patch = mpatch.Polygon(vertices, fc=roof_color, ec='k')
                patches.append(new_patch)
            if layer[0] == 'c':
                new_patch = mpatch.Rectangle((x+0.25*w_unit,l_id*h_unit+y),w_unit,h_unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
            if layer[2] == 'c':
                new_patch = mpatch.Rectangle((x+1.75*w_unit,l_id*h_unit+y),w_unit,h_unit,
                                             fc=cube_color, ec='k')
                patches.append(new_patch)
        return patches

    N = 15
    w = 1.0
    h = 1.0
    offset = 0
    i = 0

    plt.figure(dpi=300)
    ax = plt.gca()

    for max_height in [5,4,3, 2, 1]:
        space = StructuredGoalSpace(max_height)
        for struct in space.terminal_goals:
            x = (i % N) * w
            y = (i // N) * h + offset
            [ax.add_patch(p) for p in add_structure(np.array((x,y)), struct)]
            i += 1
        if i%N != 0:
            i += N - (i % N)
        offset += 0.5

    plt.axis('off')
    plt.xlim((-0.1, N*w+0.1))
    plt.ylim((-0.1, y+h+0.1))
    plt.tight_layout()

    plt.savefig('goals_by_height.png')
    # plt.show()

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatch
    # import seaborn as sns
    # show_all()
    # exit()
    for i in range(1,6):
        print(i, len(StructuredGoalSpace(i)))

    exit()
    space = StructuredGoalSpace(5)
    l = []
    for g in space.terminal_goals:
        l.append(space.opt_dist(g))
    print(sum(l)/len(l))
    exit()

    for i in range(1,6):
        space = StructuredGoalSpace(i)
        print(len(space.terminal_goals))
        # plot_structures(space.terminal_goals)
    plt.show()
    exit()
    # print(len(space.all_structures))
    space.add_cylinder_object()
    # print(space.terminal_goals)
    # print(space.novel_structures)
    exit()
    for k,v in space.structure_dict.items():
        print(k, v)
    exit()
    space.add_cylinder_object()
    for g in space.terminal_goals:
        print(g, space.paths[g])
    exit()
    for i in [1,2,3,4,5]:
        space = StructuredGoalSpace(i)
        n_goals = len(space.goals)
        n_term = len(space.terminal_goals)
        print(f'{i} : {n_goals}, {n_term}')
    exit()
    space.constrain_to('_b_,c_c,_r_')
    for k, g in space.paths.items():
        print(k, g)
    # space = StructuredGoalSpace(3)
    # for goal, path in space.paths.items():
        # print(goal, path)
    # exit()
    # print(len(space.terminal_goals))
    # for t in space.terminal_goals:
        # print(t)
    # exit()
    # plot_structures(terminal_structures)
