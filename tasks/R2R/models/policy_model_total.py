import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import wandb

from models.modules import build_mlp, SoftAttention, PositionalEncoding, create_new_mask, proj_masking, Dynamic_conv2d
from models.policy_model_partial import HighLevelModel, LowLevelModel

class HybridAgent_high(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, batch_size, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable, max_len, fc_bias=True):
        super(HybridAgent_high, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opts = opts
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        if opts.action_embedding:
            self.action_embedding = nn.Embedding(opts.max_navigable + 1, action_embedding_size, padding_idx=opts.max_navigable)  # 0~35: moving, 36: <STAY>

        # Only using high level visual feature
        assert opts.model_state == 1, 'Check model state'
        
        self.high_model = HighLevelModel(opts, batch_size, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                                        rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable=max_navigable)

        self.h0_t_ctx_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)
        self.soft_attn = SoftAttention()
        self.dropout = nn.Dropout(p=rnn_dropout)
        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1

    def forward(self, model_input,  navigable_index=None, ctx_mask=None, input_type=None):
        """ 
        [Model input]

        img_feat: batch x 36 x img_feature_size
        depth_feat: None
        obj_detection_feat: None
        num_navigable_feat: batch x 36
        navigable_index: list of list
        pre_action: previous attended action feature, list of length same as batch size
        pre_action_feat(= previous attended action feature): batch x action_feature_size  (if action_embedding==True)
        pre_action_feat(= previous attended action feature): batch x (max_navigable + 1)  # 0~35: moving, 36: <STAY>  (if action_embedding==False)
        h_0, c_0: batch x rnn_dim
        ctx: batch x seq_len x rnn_dim
        weighted_ctx: batch x rnn_hiddin_size
        ctx_mask: batch x seq_len - indices to be masked
        """

        assert input_type in [None, 'ctx']

        if input_type == 'ctx':
            h_0, ctx = model_input
            positioned_ctx = self.lang_position(ctx)
            weighted_ctx, ctx_attn = self.soft_attn(self.h0_t_ctx_fc(h_0), positioned_ctx, mask=ctx_mask)
            
            return weighted_ctx, ctx_attn

        else:
            img_feat, _, _, num_navigable_feat, pre_action, h_0, c_0, weighted_ctx, ctx_attn  = model_input

            if self.opts.action_embedding:
                # action embedding
                pre_action_feat = self.action_embedding(pre_action)
            else:
                # action one-hot encoding
                assert pre_action.shape[-1] != 1, 'Check action index shape'
                pre_action_feat = torch.zeros((pre_action.shape[0], self.opts.max_navigable+1), requires_grad=False).to(self.device)
                pre_action_feat.scatter_(1, pre_action.unsqueeze(-1), 1)

            high_h_1, high_c_1, high_visual_feat, high_img_attn, proj_navigable_feat, high_navigable_mask \
                = self.high_model((img_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')

            navigable_mask = high_navigable_mask

            # update history state
            h_1 = high_h_1
            c_1 = high_c_1

            # policy network
            high_logit = self.high_model((proj_navigable_feat, h_1, weighted_ctx), input_type='action')
            logit = high_logit

            # value estimation
            concat_value_input = self.h2_fc_lstm(torch.cat((h_0, high_visual_feat), 1))  # HOW ABOUT USING BOTH HIGH & LOW VISUAL FEAT?
            h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))
            value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

            low_visual_feat = None

            return h_1, c_1, weighted_ctx, high_img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask

class HybridAgent_low(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, batch_size, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable, max_len, fc_bias=True):
        super(HybridAgent_low, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opts = opts
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        if opts.action_embedding:
            self.action_embedding = nn.Embedding(opts.max_navigable + 1, action_embedding_size, padding_idx=opts.max_navigable)  # 0~35: moving, 36: <STAY>

        # Only using low level visual feature
        assert opts.model_state == 2, 'Check model state'
        
        self.low_model = LowLevelModel(rnn_hidden_size, action_embedding_size, \
                                        rnn_dropout, max_navigable=max_navigable, opts=opts)

        self.h0_t_ctx_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)
        self.soft_attn = SoftAttention()
        self.dropout = nn.Dropout(p=rnn_dropout)
        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.low_visual_feat_projection = nn.Linear(max_navigable, img_fc_dim[-1], bias=fc_bias)
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1

    def forward(self, model_input,  navigable_index=None, ctx_mask=None, input_type=None):
        """ 
        [Model input]

        img_feat: None
        depth_feat[0](= normalized_raw_depth_feat): batch x 36 x image_h x image_w
        depth_feat[1](= normalized_clip_depth_feat): batch x 36 x image_h x image_w
        obj_detection_feat: batch x 36 x image_h x image_w
        num_navigable_feat: batch x 36
        navigable_index: list of list
        pre_action: previous attended action feature, list of length same as batch size
        pre_action_feat(= previous attended action feature): batch x action_feature_size  (if action_embedding==True)
        pre_action_feat(= previous attended action feature): batch x (max_navigable + 1)  # 0~35: moving, 36: <STAY>  (if action_embedding==False)
        h_0, c_0: batch x rnn_dim
        ctx: batch x seq_len x rnn_dim
        weighted_ctx: batch x rnn_hiddin_size
        ctx_mask: batch x seq_len - indices to be masked
        """

        assert input_type in [None, 'ctx']

        if input_type == 'ctx':
            h_0, ctx = model_input
            positioned_ctx = self.lang_position(ctx)
            weighted_ctx, ctx_attn = self.soft_attn(self.h0_t_ctx_fc(h_0), positioned_ctx, mask=ctx_mask)
            
            return weighted_ctx, ctx_attn

        else:
            _, depth_feat, obj_detection_feat, num_navigable_feat, pre_action, h_0, c_0, weighted_ctx, ctx_attn  = model_input

            if self.opts.action_embedding:
                # action embedding
                pre_action_feat = self.action_embedding(pre_action)
            else:
                # action one-hot encoding
                assert pre_action.shape[-1] != 1, 'Check action index shape'
                pre_action_feat = torch.zeros((pre_action.shape[0], self.opts.max_navigable+1), requires_grad=False).to(self.device)
                pre_action_feat.scatter_(1, pre_action.unsqueeze(-1), 1)

            low_h_1, low_c_1, low_visual_feat, low_navigable_mask \
                = self.low_model((depth_feat, obj_detection_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')
            
            navigable_mask = low_navigable_mask

            # update history state
            h_1 = low_h_1
            c_1 = low_c_1

            # policy network
            low_logit = self.low_model(low_visual_feat, input_type='action')
            logit = low_logit

            # value estimation
            concat_value_input = self.h2_fc_lstm(torch.cat((h_0, self.low_visual_feat_projection(low_visual_feat)), 1))
            h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))  #2 #2
            value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

            high_img_attn = None

            return h_1, c_1, weighted_ctx, high_img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask

class HybridAgent_total(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, batch_size, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable, max_len, fc_bias=True):
        super(HybridAgent_total, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opts = opts
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        if opts.action_embedding:
            self.action_embedding = nn.Embedding(opts.max_navigable + 1, action_embedding_size, padding_idx=opts.max_navigable)  # 0~35: moving, 36: <STAY>

        assert opts.model_state == 3, 'Check model state'
        # Using both high level and low level visual feature
        self.hybrid_weight = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        
        self.high_model = HighLevelModel(opts, batch_size, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                                        rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable=max_navigable)
        self.low_model = LowLevelModel(rnn_hidden_size, action_embedding_size, \
                                        rnn_dropout, max_navigable=max_navigable, opts=opts)

        self.h0_t_ctx_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)
        self.soft_attn = SoftAttention()
        self.dropout = nn.Dropout(p=rnn_dropout)
        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.low_visual_feat_projection = nn.Linear(max_navigable, img_fc_dim[-1], bias=fc_bias)
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1

    def forward(self, model_input,  navigable_index=None, ctx_mask=None, input_type=None):
        """ 
        [Model input]

        img_feat: batch x 36 x img_feature_size
        depth_feat[0](= normalized_raw_depth_feat): batch x 36 x image_h x image_w
        depth_feat[1](= normalized_clip_depth_feat): batch x 36 x image_h x image_w
        obj_detection_feat: batch x 36 x image_h x image_w
        num_navigable_feat: batch x 36
        navigable_index: list of list
        pre_action: previous attended action feature, list of length same as batch size
        pre_action_feat(= previous attended action feature): batch x action_feature_size  (if action_embedding==True)
        pre_action_feat(= previous attended action feature): batch x (max_navigable + 1)  # 0~35: moving, 36: <STAY>  (if action_embedding==False)
        h_0, c_0: batch x rnn_dim
        ctx: batch x seq_len x rnn_dim
        weighted_ctx: batch x rnn_hiddin_size
        ctx_mask: batch x seq_len - indices to be masked
        """

        assert input_type in [None, 'ctx']

        if input_type == 'ctx':
            h_0, ctx = model_input
            positioned_ctx = self.lang_position(ctx)
            weighted_ctx, ctx_attn = self.soft_attn(self.h0_t_ctx_fc(h_0), positioned_ctx, mask=ctx_mask)
            
            return weighted_ctx, ctx_attn

        else:
            img_feat, depth_feat, obj_detection_feat, num_navigable_feat, pre_action, h_0, c_0, weighted_ctx, ctx_attn  = model_input

            hybrid_weight = F.softmax(self.hybrid_weight, dim=0)

            if self.opts.action_embedding:
                # action embedding
                pre_action_feat = self.action_embedding(pre_action)
            else:
                # action one-hot encoding
                assert pre_action.shape[-1] != 1, 'Check action index shape'
                pre_action_feat = torch.zeros((pre_action.shape[0], self.opts.max_navigable+1), requires_grad=False).to(self.device)
                pre_action_feat.scatter_(1, pre_action.unsqueeze(-1), 1)

            high_h_1, high_c_1, high_visual_feat, high_img_attn, proj_navigable_feat, high_navigable_mask \
                = self.high_model((img_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')

            low_h_1, low_c_1, low_visual_feat, low_navigable_mask \
                = self.low_model((depth_feat, obj_detection_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')
            
            assert (high_navigable_mask == low_navigable_mask).all()
            navigable_mask = high_navigable_mask

            # update history state
            h_1 = (high_h_1 * hybrid_weight[0]) + (low_h_1 * hybrid_weight[1])
            c_1 = (high_c_1 * hybrid_weight[0]) + (low_c_1 * hybrid_weight[1])

            # policy network
            high_logit = self.high_model((proj_navigable_feat, h_1, weighted_ctx), input_type='action')
            low_logit = self.low_model(low_visual_feat, input_type='action')
            logit = (high_logit * hybrid_weight[0]) + (low_logit * hybrid_weight[1])

            # value estimation
            total_visual_feat = (high_visual_feat * hybrid_weight[0]) + (self.low_visual_feat_projection(low_visual_feat) * hybrid_weight[1])
            concat_value_input = self.h2_fc_lstm(torch.cat((h_0, total_visual_feat), 1))
            h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))
            value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

            ##### Logging #####
            if self.opts.wandb_visualize:
                wandb.log({'hybrid_weight': wandb.Histogram(hybrid_weight.detach().cpu().numpy())})

            return h_1, c_1, weighted_ctx, high_img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask