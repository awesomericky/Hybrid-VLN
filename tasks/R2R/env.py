''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from os.path import join
import pickle
import torch
from scipy.special import softmax

from utils import load_datasets, load_nav_graphs

csv.field_size_limit(sys.maxsize)

PREPROCESSED_DATA_FOLDER = 'preprocessed_data'
SAFE_DEPTH_THRESHOLD = 5*4000
INSTRUCTION_ATTENTION_THRESHOLD = 5


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        if feature_store:
            print('Loading image features from %s' % feature_store)
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    # self.image_h = int(item['image_h'])
                    # self.image_w = int(item['image_w'])
                    # self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    self.features[long_id] = np.frombuffer(base64.b64decode(item['features']),
                            dtype=np.float32).reshape((36, 2048))
        else:
            print('Image features not provided')
            self.features = None
        
        # downsize image (downsized by 4)
        self.image_w = 160  # 640/4
        self.image_h = 120  # 480/4
        self.vfov = 60

        # Rendering is not needed because needed features(ResNet image, depth, object detection, number of navigable locations)
        # are precomputed and will be loaded from memory
        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        # self.sim.setDepthEnabled(True)
        # self.sim.setPreloadingEnabled(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCacheSize(self.batch_size*2)
        self.sim.initialize()

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0]*self.batch_size)

    def getStates(self, instruction_datas):

        """

        1) State (type & dimension)
        - ResNet image(rgb) feature : (36, feature_dim)
        - Depth (normalized free space) : (2, single_view_height * 3, single_view_width * 12)
        - Object detection : (1, single_view_height * 3, single_view_width * 12)
        - Number of navigable locations (normalized) : (1, 3, 12)

        (*1st channel in depth: normalized_raw_depth, 2nd channel in depth: normalized_clip_depth)

        2) Spatial meaning in state dimention
         ㅡ                                                 ㅡ
        |   24  25  26  27  28  29  30  31  32  33  34  35   |   (looking up)
        |                                                    |
        |   12  13  14  15  16  17  18  19  20  21  22  23   |   (looking at horizon)
        |                                                    |
        |    0   1   2   3   4   5   6   7   8   9  10  11   |   (looking down)
         ㅡ                                                 ㅡ

        """

        total_feature_states = []
        states = self.sim.getState()
        for state in states:
            scanId = state.scanId
            viewpointId = state.location.viewpointId

            # ResNet image(rgb) feature
            long_id = self._make_id(scanId, viewpointId)
            if self.features:
                spatial_img_feature = self.features[long_id]
            else:
                spatial_img_feature = None
            
            # Depth (normalized free space)
            spatial_depth = np.zeros((2, self.image_h*3, self.image_w*12), dtype=np.float32)
            file_name = viewpointId + '.npz'
            file_name = join(join(PREPROCESSED_DATA_FOLDER, 'depth_result'), join(scanId, file_name))

            raw_depth = np.load(file_name)['depth'] # (1, 120*3, 160*12)
            normalized_raw_depth = raw_depth/np.max(raw_depth)
            clip_depth = np.clip(raw_depth, a_max=SAFE_DEPTH_THRESHOLD)
            normalized_clip_depth = clip_depth/np.max(clip_depth)

            spatial_depth[0, :, :] = normalized_raw_depth
            spatial_depth[1, :, :] = normalized_clip_depth

            # Object detection
            spatial_obj_detection = np.zeros((1, self.image_h*3, self.image_w*12), dtype=np.float32)

            instruction_data = instruction_datas[states.index(state)]
            instruction, instruction_attn = instruction_data  # instruction_data = (instruction, instruction_attn) # 'instruction' type: list, 'instruction_attn' type: tensor
            instruction_attn_values, instruction_attn_indices = torch.topk(instruction_attn, INSTRUCTION_ATTENTION_THRESHOLD)

            instruction_attn_values = instruction_attn_values.detach().cpu().numpy()
            instruction_attn_indices = instruction_attn_indices.detach().cpu().numpy()
            attn_words = []
            for instruction_attn_indice in instruction_attn_indices:
                attn_words.append(instruction[instruction_attn_indice])

            for i in range(36):
                file_name = viewpointId + '_' + str(i) + '.pickle'
                file_name = join(join(PREPROCESSED_DATA_FOLDER, 'obj_detection_result'), join(scanId, file_name))

                with open(file_name, 'rb') as f:
                    object_detection_result = pickle.load(f)
                
                detected_objects = object_detection_result.keys()

                final_attn_words = []
                final_attn_values = []
                final_attn_bboxs = []

                for detected_object in detected_objects:
                    if detected_object in attn_words:
                        final_attn_words.append(detected_object)
                        final_attn_values.append(instruction_attn_values[attn_words.index(detected_object)])
                        final_attn_bboxs.append(object_detection_result[detected_object])
                
                if len(final_attn_words) == 0:
                    final_partial_obj_detection = np.zeros((1, self.image_h, self.image_w), dtype=np.float32)
                else:
                    partial_obj_detection = np.zeros((len(final_attn_words), self.image_h, self.image_w), dtype=np.float32)
                    
                    obj_weight = softmax(np.asarray(final_attn_values))
                    obj_weight = obj_weight[:, np.newaxis, np.newaxis]

                    for final_attn_bbox in final_attn_bboxs:
                        index = final_attn_bboxs.index(final_attn_bbox)
                        width = round(final_attn_bbox[2] - final_attn_bbox[0])
                        height = round(final_attn_bbox[3] - final_attn_bbox[1])
                        x = round(final_attn_bbox[0])
                        y = self.image_h - round(final_attn_bbox[1])

                        partial_obj_detection[index, y-height:y, x:x+width] = 1
                    
                    final_partial_obj_detection = np.sum(partial_obj_detection*obj_weight, axis=0)
                
                y = i // 12
                x = i % 12
                spatial_obj_detection[0, self.image_h*(2-y):self.image_h*(3-y), self.image_w*x:self.image_w*(x+1)] = final_partial_obj_detection
            
            # Number of navigable locations # (normalized)
            file_name = viewpointId + '.npz'
            file_name = join(join(PREPROCESSED_DATA_FOLDER, 'navigable_result'), join(scanId, file_name))
            spatial_n_navigable = np.load(file_name)['navigable']  # (1, 3, 12)
            # spatial_n_navigable = spatial_n_navigable/np.max(spatial_n_navigable)
            
            total_feature_states.append((spatial_img_feature, spatial_depth, spatial_obj_detection, spatial_n_navigable, state))
        
        return total_feature_states
    
    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        ix = []
        heading = []
        elevation = []
        for i,h,e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        actions = []
        for i, index in enumerate(simple_indices):
            if index == 0:
                actions.append((1, 0, 0))
            elif index == 1:
                actions.append((0,-1, 0))
            elif index == 2:
                actions.append((0, 1, 0))
            elif index == 3:
                actions.append((0, 0, 1))
            elif index == 4:
                actions.append((0, 0,-1))
            else:
                sys.exit("Invalid simple action")
        self.makeActions(actions)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, opts, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.data = []
        self.scans = []
        self.opts = opts

        for item in load_datasets(splits):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        pos = [state.location.x, state.location.y, state.location.z]
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - pos
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right
    
    def _pano_navigable(self, state, goalViewpointId):
        """ Get the navigable viewpoints and their relative heading and elevation,
            as well as the index for 36 image features. """
        navigable_graph = self.graphs[state.scanId].adj[state.location.viewpointId]
        teacher_path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]

        if len(teacher_path) > 1:
            next_gt_viewpoint = teacher_path[1]
        else:
            # the current viewpoint is our ground-truth
            next_gt_viewpoint = state.location.viewpointId
            gt_viewpoint_idx = (state.location.viewpointId, state.viewIndex)

        # initialize a dict to save info for all navigable points
        navigable = {}

        # add the current viewpoint into navigable, so the agent can stay
        navigable[state.location.viewpointId] = {
            'position': state.location.point,
            'heading': state.heading,
            'rel_heading': state.location.rel_heading,
            'rel_elevation': state.location.rel_elevation,
            # 'index': state.viewIndex
            'index': 36  # 0~35 correspind to moving, 36 correspond to stop
        }

        index_history = []

        for viewpoint_id, weight in navigable_graph.items():
            dict_tmp = {}

            node = self.graphs[state.scanId].nodes[viewpoint_id]
            curr_point = [state.location.x, state.location.y, state.location.z]
            target_rel = node['position'] - curr_point
            dict_tmp['position'] = list(node['position'])

            # note that this "heading" is computed regarding the global coordinate
            # the actual "heading" between the current viewpoint to next viewpoint
            # needs to take into account the current heading
            target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
            if target_heading < 0:
                target_heading += 2.0*math.pi

            assert state.heading >= 0

            dict_tmp['rel_heading'] = target_heading - state.heading
            dict_tmp['heading'] = target_heading

            # compute the relative elevation
            dist = math.sqrt(sum(target_rel ** 2))  # compute the relative distance
            rel_elevation = np.arcsin(target_rel[2] / dist)
            dict_tmp['rel_elevation'] = rel_elevation

            # elevation level -> 0 (bottom), 1 (middle), 2 (top)
            elevation_level = round(rel_elevation / (30 * math.pi / 180)) + 1
            # To prevent if elevation degree > 45 or < -45
            elevation_level = max(min(2, elevation_level), 0)

            # viewpoint index depends on the elevation as well
            horizontal_idx = int(round(target_heading / (math.pi / 6.0)))  # current: -15(=345)~15, 15~45 ...
            horizontal_idx = 0 if horizontal_idx == 12 else horizontal_idx
            viewpoint_idx = int(horizontal_idx + 12 * elevation_level)

            # To check whether multiple navigable locations correspond to same index
            if viewpoint_idx in index_history:
                print('Multiple navigable locations corresponding to same index exists!!!!')
            index_history.append(viewpoint_idx)

            dict_tmp['index'] = viewpoint_idx

            # let us get the ground-truth viewpoint index for seq2seq training
            if viewpoint_id == next_gt_viewpoint:
                gt_viewpoint_idx = (viewpoint_id, viewpoint_idx)

            # save into dict
            navigable[viewpoint_id] = dict_tmp

        return navigable, gt_viewpoint_idx
    
    def shortest_path_to_gt_traj(self, state, gt_path):
        """ Compute the next viewpoint by trying to steer back to original ground truth trajectory"""
        min_steps = 100
        min_distance = 100
        current_distance = self.distances[state.scanId][state.location.viewpointId][gt_path[-1]]

        if current_distance != 0:
            for gt_viewpoint in gt_path:
                steps = len(self.paths[state.scanId][state.location.viewpointId][gt_viewpoint])
                next_distance = self.distances[state.scanId][gt_viewpoint][gt_path[-1]]

                # if the next viewpoint requires moving and its distance to the goal is closer
                if steps > 0 and next_distance < current_distance:
                    if min_steps >= steps and min_distance > next_distance:
                        min_steps = steps
                        min_distance = next_distance
                        next_viewpoint = gt_viewpoint
        else:
            next_viewpoint = state.location.viewpointId
        return next_viewpoint

    def _get_obs(self, instruction_datas):
        obs = []
        for i,(spatial_img_feature, spatial_depth, spatial_obj_detection, spatial_n_navigable, state) in enumerate(self.env.getStates(instruction_datas)):
            item = self.batch[i]

            if self.opts.follow_gt_traj:
                goal_viewpoint = self.shortest_path_to_gt_traj(state, item['path'])
            else:
                goal_viewpoint = item['path'][-1]

            # compute the navigable viewpoints and next ground-truth viewpoint
            navigable, gt_viewpoint_idx = self._pano_navigable(state, goal_viewpoint)

            # in synthetic data, path_id is unique since we only has 1 instruction per path, we will then use it as 'instr_id'
            if 'synthetic' in self.splits:
                assert len(self.splits) == 1 'Should only use synthetic data, when using synthetic data for traing'
                item['instr_id'] = str(item['path_id'])
            
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'spatial_image_feature': spatial_image_feature,
                'spatial_depth': spatial_depth,
                'spatial_obj_detection': spatial_obj_detection,
                'spatial_n_navigable': spatial_n_navigable,
                'step' : state.step,
                'navigableLocations': navigable,
                'instructions' : item['instructions'],
                'teacher': item['path'],
                # 'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'new_teacher': self.paths[state.scanId][state.location.viewpointId][item['path'][-1]],
                'gt_viewpoint_idx': gt_viewpoint_idx
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, scanIds, viewpointIds, headings):
        ''' Take action (same interface as makeActions) '''
        # self.env.makeActions(actions)
        # return self._get_obs()

        def rotate_to_target_heading(target_heading, state):
            if target_heading < 0:
                target_heading += 2.0 * math.pi
            if abs(target_heading - state.heading) * 180 / math.pi < 15 or abs(target_heading - state.heading) * 180 / math.pi > 345:  # if the target relative heading is less than 15 degree, stop rotating
                return (0, 0, 0)
            if state.heading > target_heading and state.heading - target_heading < math.pi:
                return (0, -1, 0)  # Turn left
            if target_heading > state.heading and target_heading - state.heading > math.pi:
                return (0, -1, 0)  # Turn left
            return (0, 1, 0)  # Turn right

        if self.opts.teleporting:
            self.env.newEpisodes(scanIds, viewpointIds, headings)
        else:
            for i in range(self.batch_size):
                action = None
                # move the agent to the target viewpoint internally, instead of directly 'teleporting'
                while action != (0, 0, 0):
                    state = self.env.sim.getState()[i]
                    action = self._shortest_path_action(state, viewpointIds[i])
                    index, heading, elevation = action

                    index_s = [0] * self.batch_size
                    heading_s = [0] * self.batch_size
                    elevation_s = [0] * self.batch_size

                    index_s[i] = index
                    heading_s[i] = heading
                    elevation_s[i] = elevation

                    self.env.sim.makeAction(index_s, heading_s, elevation_s)
                # assert state.location.viewpointId == viewpointIds[i], 'the actions took internally was not correct.'

                action = None
                # we have reached the viewpoint, now let's rotate to the corresponding heading
                while action != (0, 0, 0):
                    state = self.env.sim.getState()[i]
                    action = rotate_to_target_heading(headings[i], state)
                    index, heading, elevation = action

                    index_s = [0] * self.batch_size
                    heading_s = [0] * self.batch_size
                    elevation_s = [0] * self.batch_size

                    index_s[i] = index
                    heading_s[i] = heading
                    elevation_s[i] = elevation

                    self.env.sim.makeAction(index_s, heading_s, elevation_s)

        return self._get_obs()
