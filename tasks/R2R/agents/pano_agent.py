import json
import random
import numpy as np
import pdb
import copy

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import padding_idx

torch.autograd.set_detect_anomaly = True

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
            m_action = action.detach()
        elif self.feedback == 'sample':
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
            if self.opts.rl_weight != 0:
                self.rl_data.action_log_probs.append(m.log_prob(action))
            m_action = action.clone()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to self.stop_idx if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    m_action[i] = self.stop_idx
                    pass
        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended, max_navigable):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] < max_navigable:
                next_viewpoint_idx.append(int(action[i]))
           
                curr_navigable_index = np.asarray(navigable_index[i])
                corresponding_index = list(np.where(curr_navigable_index == action[i].numpy())[0])
                choice_index = np.random.choice(corresponding_index, 1)[0]
                next_viewpoints.append(viewpoints[i][choice_index+1])  # add 1 to index considering the <STAY> viewpoint in 0 index of viewpooints, which is not considered in navigable_index
            else:
                next_viewpoint_idx.append('STAY')
                next_viewpoints.append(viewpoints[i][0])
                assert ob['viewpoint'] == next_viewpoints[-1]
                ended[i] = True
            
            next_headings.append(ob['navigableLocations'][next_viewpoints[-1]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended):
        # Get the img_feat, depth_feat, obj_detection_feat, num_navigable_feat
        """
        [Feature dimension]

        img_feat: batch x 36 x img_feature_size
        normalized_raw_depth_feat: batch x 36 x image_h x image_w
        normalized_clip_depth_feat: batch x 36 x image_h x image_w
        obj_detection_feat: batch x 36 x image_h x image_w
        num_navigable_feat: batch x 36
        """
        current_mode = ''

        if obs[0]['spatial_image_feature'] is not None:
            img_feat_shape = obs[0]['spatial_image_feature'].shape
            img_feat = torch.zeros(len(obs), img_feat_shape[0], img_feat_shape[1])
            current_mode += 'a'
        else:
            img_feat = None

        if obs[0]['spatial_depth'] is not None:
            normalized_raw_depth_feat_shape = obs[0]['spatial_depth']['normalized_raw_depth'].shape
            normalized_clip_depth_feat_shape = obs[0]['spatial_depth']['normalized_clip_depth'].shape
            normalized_raw_depth_feat = torch.zeros(len(obs), normalized_raw_depth_feat_shape[0], normalized_raw_depth_feat_shape[1], normalized_raw_depth_feat_shape[2])
            normalized_clip_depth_feat = torch.zeros(len(obs), normalized_clip_depth_feat_shape[0], normalized_clip_depth_feat_shape[1], normalized_clip_depth_feat_shape[2])
            current_mode += 'b'
        else:
            depth_feat = None

        if obs[0]['spatial_obj_detection'] is not None:
            obj_detection_feat_shape = obs[0]['spatial_obj_detection'].shape
            obj_detection_feat = torch.zeros(len(obs), obj_detection_feat_shape[0], obj_detection_feat_shape[1], obj_detection_feat_shape[2])
            current_mode += 'c'
        else:
            obj_detection_feat = None

        if obs[0]['spatial_n_navigable'] is not None:
            num_navigable_feat_shape = obs[0]['spatial_n_navigable'].shape
            num_navigable_feat = torch.zeros(len(obs), num_navigable_feat_shape[0])
            current_mode += 'd'
        else:
            num_navigable_feat = None
        
        assert current_mode in ['ad', 'bcd', 'abcd'], 'Check the env.py code'

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            if current_mode == 'ad':
                img_feat[i, :] = torch.from_numpy(ob['spatial_image_feature'])
                num_navigable_feat[i, :] = torch.from_numpy(ob['spatial_n_navigable'])
            elif current_mode == 'bcd':
                normalized_raw_depth_feat[i, :] = torch.from_numpy(ob['spatial_depth']['normalized_raw_depth'])
                normalized_clip_depth_feat[i, :] = torch.from_numpy(ob['spatial_depth']['normalized_clip_depth'])
                depth_feat = [normalized_raw_depth_feat, normalized_clip_depth_feat]
                obj_detection_feat[i, :] = torch.from_numpy(ob['spatial_obj_detection'])
                num_navigable_feat[i, :] = torch.from_numpy(ob['spatial_n_navigable'])
            else:  # current_mode == 'abcd'
                img_feat[i, :] = torch.from_numpy(ob['spatial_image_feature'])
                normalized_raw_depth_feat[i, :] = torch.from_numpy(ob['spatial_depth']['normalized_raw_depth'])
                normalized_clip_depth_feat[i, :] = torch.from_numpy(ob['spatial_depth']['normalized_clip_depth'])
                depth_feat = [normalized_raw_depth_feat, normalized_clip_depth_feat]
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

        # Just for saving rl data and gradient path
        self.rl_data = RL_data()

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

                if img_attn is not None:
                    traj[i]['img_attn'].append(img_attn[i].detach().cpu().numpy().tolist())
                else:
                    traj[i]['img_attn'].append(None)
                if low_visual_feat is not None:
                    traj[i]['low_visual_feat'].append(low_visual_feat[i].detach().cpu().numpy().tolist())
                else:
                    traj[i]['low_visual_feat'].append(None)

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
        obs, past_distance = np.array(self.env.reset())  # load a mini-batch
        instructions = []

        for ob in obs:
           instructions.append(ob['instr_decoding'])

        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)

        pre_action = self.stop_idx * torch.ones(batch_size)
        pre_action = pre_action.long().to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0
        for step in range(self.opts.max_episode_len):
            weighted_ctx, ctx_attn = self.model((h_t, ctx), ctx_mask=ctx_mask, input_type='ctx')

            instruction_datas = [instructions, ctx_attn]
            obs, _ = self.env._get_obs(model_input=instruction_datas)

            img_feat, depth_feat, obj_detection_feat, num_navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            if self.opts.model_state == 1:
                # Only using high level visual feature
                img_feat = img_feat.to(self.device)
                num_navigable_feat = num_navigable_feat.to(self.device)
                target = torch.LongTensor(target_index).to(self.device)
            elif self.opts.model_state == 2:
                # Only using low level visual feature
                depth_feat[0] = depth_feat[0].to(self.device)  # normalized_raw_depth_feat
                depth_feat[1] = depth_feat[1].to(self.device)  # normalized_clip_depth_feat
                obj_detection_feat = obj_detection_feat.to(self.device)
                num_navigable_feat = num_navigable_feat.to(self.device)
                target = torch.LongTensor(target_index).to(self.device)
            else:
                # Using both high level and low level visual feature
                img_feat = img_feat.to(self.device)
                depth_feat[0] = depth_feat[0].to(self.device)  # normalized_raw_depth_feat
                depth_feat[1] = depth_feat[1].to(self.device)  # normalized_clip_depth_feat
                obj_detection_feat = obj_detection_feat.to(self.device)
                num_navigable_feat = num_navigable_feat.to(self.device)
                target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, weighted_ctx, img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask = self.model(
                (img_feat, depth_feat, obj_detection_feat, num_navigable_feat,
                pre_action, h_t, c_t, weighted_ctx, ctx_attn),  navigable_index=navigable_index, ctx_mask=ctx_mask)

            navigable_mask = torch.cat((navigable_mask, self.stop_idx_mask), dim=1)

            # To avoid NaN when multiply prob and logprob, we clone the logit and perform masking
            logit_for_logprob = logit.clone()
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            logit_for_logprob.data.masked_fill_((navigable_mask == 0).data, -float('1e7'))

            action_prob = F.softmax(logit, dim=1)
            action_logprob = F.log_softmax(logit_for_logprob, dim=1)
            action_logprob = action_logprob * navigable_mask

            # Compute IL loss & Entorpy loss
            entropy_loss = torch.sum(action_prob * action_logprob, dim=1, keepdim=True).mean()
            current_logit_loss = self.criterion(logit, target)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss + self.opts.entropy_weight * entropy_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + \
                                    (1 - self.opts.value_loss_weight) * current_logit_loss + \
                                    self.opts.entropy_weight * entropy_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            
            loss += current_loss

            # """ Practice """
            # logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            # action_prob = F.softmax(logit, dim=1)
            # current_val_loss = self.get_value_loss_from_start(traj, value, ended)
            # self.value_loss += current_val_loss
            # loss += current_val_loss
            # """ Practice """

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(action_prob, ended, is_prob=True, fix_action_ended=self.opts.fix_action_ended)

            prev_ended = copy.deepcopy(ended)
            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended, self.opts.max_navigable)

            # make a viewpoint change in the env
            obs, current_distance = self.env.step(scan_id, next_viewpoints, next_headings)

            # Compute reward and probability for RL loss (continuous reward: distance change toward goal / sparse reward: success or not)
            if self.opts.rl_weight != 0 and self.feedback is not 'argmax':
                current_reward = np.array(past_distance) - np.array(current_distance)
                dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]
                success_reward = np.array(np.array(dist_from_goal) < 3.0, dtype=int) * \
                                    np.array(ended, dtype=int) * (1 - np.array(prev_ended, dtype=int))
                current_reward += success_reward

                self.rl_data.rewards.append(current_reward)
                past_distance = current_distance

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, low_visual_feat, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_action = action.long().to(self.device)

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        # compute RL loss
        if self.opts.rl_weight != 0 and self.feedback is not 'argmax':
            exp_return = 0
            returns = []
            rl_losses = []
            for reward in self.rl_data.rewards[::-1]:
                exp_return = reward + self.opts.rl_discount_factor * exp_return
                returns.insert(0, exp_return)
            for idx, _reward in enumerate(returns):
                rl_losses.append(torch.tensor(np.array(_reward, dtype=np.float32)) * self.rl_data.action_log_probs[idx])
            final_rl_loss = -torch.sum(torch.cat([rl_loss.unsqueeze(-1) for rl_loss in rl_losses], dim=1), keepdim=True, dim=1).mean().to(self.device)
            loss += self.opts.rl_weight * final_rl_loss

            del self.rl_data.action_log_probs[:]
            del self.rl_data.rewards[:]

        return loss, traj

class RL_data(object):
    def __init__(self):
        self.action_log_probs = []
        self.rewards = []