import json
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import padding_idx

class PanoBaseAgent(object):
    """ Base class for an R2R agent with panoramic view and action. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
    
    def write_results(self):
        output = []
        for k, v in self.results.items():
            output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path'],
                    'distance': v['distance'],
                    'img_attn': v['img_attn'],
                    'low_visual_feat': v['low_visual_feat'],
                    'ctx_attn': v['ctx_attn'],
                    'value': v['value'],
                    'viewpoint_idx': v['viewpoint_idx'],
                    'navigable_idx': v['navigable_idx']
                }
            )
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    
    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['instr_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['instr_id'])]
        distance = self.env.distances[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _select_action(self, logit, ended, is_prob=False, fix_action_ended=True):
        logit_cpu = logit.clone().cpu()
        if is_prob:
            probs = logit_cpu
        else:
            probs = F.softmax(logit_cpu, 1)

        if self.feedback == 'argmax':
            _, action = probs.max(1)  # student forcing - argmax
            action = action.detach()
        elif self.feedback == 'sample':
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to self.stop_idx if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    action[i] = self.stop_idx

        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended, max_navigable):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] < max_navigable:
                next_viewpoint_idx.append(action[i])
           
                curr_navigable_index = np.asarray(navigable_index[i])
                corresponding_index = list(np.where(curr_navigable_index == action[i].numpy())[0])
                choice_index = np.random.choice(corresponding_index, 1)[0]
                next_viewpoints.append(viewpoints[i][choice_index+1])  # add 1 to index considering the <STAY> viewpoint in 0 index of viewpooints, which is not considered in navigable_index
            else:
                next_viewpoint_idx.append('STAY')
                next_viewpoints.append(viewpoints[i][0])
                assert ob['viewpoint'] == next_viewpoints[-1]
                ended[i] = True
            
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended):
        # Get the img_feat, depth_feat, obj_detection_feat, num_navigable_feat
        """
        [Feature dimension]

        img_feat: batch x 36 x img_feature_size
        depth_feat: batch x 2 x (image_h*3) x (image_w*12)
        obj_detection_feat: batch x 1 x (image_h*3) x (image_w*12)
        num_navigable_feat: batch x 3 x 12
        """

        img_feat_shape = obs[0]['spatial_image_feature'].shape
        depth_feat_shape = obs[0]['spatial_depth'].shape
        obj_detection_feat_shape = obs[0]['spatial_obj_detection'].shape
        num_navigable_feat_shape = obs[0]['spatial_n_navigable'].shape

        img_feat = torch.zeros(len(obs), img_feat_shape[0], img_feat_shape[1])
        depth_feat = torch.zeros(len(obs), depth_feat_shape[0], depth_feat_shape[1], depth_feat_shape[2])
        obj_detection_feat = torch.zeros(len(obs), obj_detection_feat_shape[0], obj_detection_feat_shape[1], obj_detection_feat_shape[2])
        num_navigable_feat = torch.zeros(len(obs), num_navigable_feat_shape[1], num_navigable_feat_shape[2])

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            img_feat[i, :] = torch.from_numpy(ob['spatial_image_feature'])
            depth_feat[i, :] = torch.from_numpy(ob['spatial_depth'])
            obj_detection_feat[i, :] = torch.from_numpy(ob['spatial_obj_detection'])
            num_navigable_feat[i, :] = torch.from_numpy(ob['spatial_n_navigable'])

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(index_list[-1])

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay" (written in policy_model.py self.stop_feat)
            assert index_list[0] == self.stop_idx
            navi_index = index_list[1: ]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

        return img_feat, depth_feat, obj_detection_feat, num_navigable_feat, (viewpoints, navigable_feat_index, target_index)

    def _sort_batch(self, obs):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)


class PanoSeq2SeqAgent(PanoBaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """
    def __init__(self, opts, env, results_path, encoder, model, feedback='sample', episode_len=20):
        super(PanoSeq2SeqAgent, self).__init__(env, results_path)
        self.opts = opts
        self.encoder = encoder
        self.model = model
        self.feedback = feedback
        self.episode_len = episode_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stop_idx = opts.max_navigable
        self.ignore_index = opts.max_navigable + 1  # we define (max_navigable+1) as ignore since 15(navigable) + 1(STOP)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.MSELoss = nn.MSELoss()
        self.MSELoss_sum = nn.MSELoss(reduction='sum')

        self.stop_idx_mask = torch.ones(opts.batch_size).unsqueeze(-1).float().to(self.device)

    def get_value_loss_from_start(self, traj, predicted_value, ended):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            value_target.append(dist_improved_from_start)

            if dist <= 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            # we will average the loss according to number of not 'ended', and use reduction='sum' for MSELoss
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def get_value_loss_from_start_sigmoid(self, traj, predicted_value, ended):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            dist_improved_from_start = 0 if dist_improved_from_start < 0 else dist_improved_from_start

            value_target.append(dist_improved_from_start)

            if dist < 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                # 'feature': [ob['feature']],
                'img_attn': [],
                'low_visual_feat': [],
                'ctx_attn': [],
                'value': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                'steps_required': [len(ob['teacher'])],
                'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)
        last_recorded = np.array([False] * batch_size)

        return traj, scan_id, ended, last_recorded

    def update_traj(self, obs, traj, img_attn, low_visual_feat, ctx_attn, value, next_viewpoint_idx,
                    navigable_index, ended, last_recorded):
        # Save trajectory output and accumulated reward
        for i, ob in enumerate(obs):
            if not ended[i] or not last_recorded[i]:
                traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                traj[i]['distance'].append(dist)
                traj[i]['img_attn'].append(img_attn[i].detach().cpu().numpy().tolist())
                traj[i]['low_visual_feat'].append(low_visual_feat[i].detach().cpu().numpy().tolist())
                traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())

                if len(value[1]) > 1:
                    traj[i]['value'].append(value[i].detach().cpu().tolist())
                else:
                    traj[i]['value'].append(value[i].detach().cpu().item())

                traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                traj[i]['navigable_idx'].append(navigable_index[i])
                traj[i]['steps_required'].append(len(ob['new_teacher']))
                self.traj_length[i] = self.traj_length[i] + 1
                last_recorded[i] = True if ended[i] else False

        return traj, last_recorded

    def rollout_hybrid(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        instructions = []

        for ob in obs:
           instructions.append(ob['instructions'])

        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)

        pre_action = self.stop_idx * torch.ones(batch_size)
        pre_action = pre_action.long().to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0
        for step in range(self.opts.max_episode_len):
            print(step)

            weighted_ctx, ctx_attn = self.model((h_t, ctx), ctx_mask=ctx_mask, input_type='ctx')

            ###1000 #500 #500 # 500
            pdb.set_trace()
            instruction_datas = [instructions, ctx_attn]
            obs = self.env._get_obs(model_input=instruction_datas)

            img_feat, depth_feat, obj_detection_feat, num_navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            #500 #0 #0
            img_feat = img_feat.to(self.device)
            depth_feat = depth_feat.to(self.device)
            obj_detection_feat = obj_detection_feat.to(self.device)
            num_navigable_feat = num_navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)
            pdb.set_trace()

            # forward pass the network
            h_t, c_t, weighted_ctx, img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask = self.model(
                (img_feat, depth_feat, obj_detection_feat, num_navigable_feat,
                pre_action, h_t, c_t, weighted_ctx, ctx_attn),  navigable_index=navigable_index, ctx_mask=ctx_mask)

            # set other values to -inf so that logsoftmax will not affect the final computed loss
            navigable_mask = torch.cat((navigable_mask, self.stop_idx_mask), dim=1)
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended, self.opts.max_navigable)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, low_visual_feat, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_action = action.long().to(self.device)

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj
