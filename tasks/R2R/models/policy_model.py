import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from models.modules import build_mlp, SoftAttention, PositionalEncoding, create_new_mask, proj_masking, Dynamic_conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridAgent(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, batch_size, training_state, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable, max_len, fc_bias=True):
        super(HybridAgent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        self.action_embedding = nn.Embedding(opts.max_navigable + 1, action_embedding_size, padding_idx=opts.max_navigable)  # 0~35: moving, 36: <STAY>

        self.high_model = HighLevelModel(opts, batch_size, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,\
                                        rnn_hidden_size, action_embedding_size, rnn_dropout, max_navigable=max_navigable)
        self.low_model = LowLevelModel(rnn_hidden_size, action_embedding_size, \
                                        rnn_dropout, max_navigable=max_navigable)

        assert training_state in [1, 2, 3]
        if training_state == 1:
            # Only using high level visual feature
            self.hybrid_weight = torch.tensor([1, 0], dtype=torch.float32, requires_grad=True, device=self.device)
        elif training_state == 2:
            # Only using low level visual feature
            self.hybrid_weight = torch.tensor([0, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        else:  # training_state == 3
            # Using both high level and low level visual feature
            self.hybrid_weight = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True, device=self.device)

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
        depth_feat: batch x 2 x (image_h*3) x (image_w*12)
        obj_detection_feat: batch x 1 x (image_h*3) x (image_w*12)
        num_navigable_feat: batch x 3 x 12
        pre_action: previous attended action feature, list of length same as batch size
        h_0, c_0: batch x rnn_dim
        ctx: batch x seq_len x rnn_dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        assert input_type in [None, 'ctx']

        if input_type == 'ctx':
            h_0, ctx = model_input  #0
            positioned_ctx = self.lang_position(ctx)  #20 #20 #20 #20
            weighted_ctx, ctx_attn = self.soft_attn(self.h0_t_ctx_fc(h_0), positioned_ctx, mask=ctx_mask)  #0
            
            return weighted_ctx, ctx_attn

        else:
            img_feat, depth_feat, obj_detection_feat, num_navigable_feat, pre_action, h_0, c_0, weighted_ctx, ctx_attn  = model_input  #0

            pre_action_feat = self.action_embedding(pre_action)  #0

            high_h_1, high_c_1, high_visual_feat, img_attn, proj_navigable_feat, high_navigable_mask \
                = self.high_model((img_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')

            low_h_1, low_c_1, low_visual_feat, low_navigable_mask \
                = self.low_model((depth_feat, obj_detection_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index), input_type='history')
            
            assert (high_navigable_mask == low_navigable_mask).all()
            navigable_mask = high_navigable_mask

            # update history state
            h_1 = (high_h_1 * self.hybrid_weight[0]) + (low_h_1 * self.hybrid_weight[1])
            c_1 = (high_c_1 * self.hybrid_weight[0]) + (low_c_1 * self.hybrid_weight[1])

            # policy network
            high_logit = self.high_model((proj_navigable_feat, h_1, weighted_ctx), input_type='action')
            low_logit = self.low_model(low_visual_feat, input_type='action')
            logit = (high_logit * self.hybrid_weight[0]) + (low_logit * self.hybrid_weight[1])

            # value estimation
            concat_value_input = self.h2_fc_lstm(torch.cat((h_0, high_visual_feat), 1))  # HOW ABOUT USING BOTH HIGH & LOW VISUAL FEAT?  #2 #0 #0
            h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))  #2 #2
            value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

            return h_1, c_1, weighted_ctx, img_attn, low_visual_feat, ctx_attn, logit, value, navigable_mask
    
class HighLevelModel(nn.Module):
    def __init__(self, opts, batch_size, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, action_embedding_size, rnn_dropout, fc_bias=True, max_navigable=36):
        super(HighLevelModel, self).__init__()

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.ctx_t_img_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

        self.soft_attn = SoftAttention(extra_mask_needed=1, batch_size=batch_size, device=self.device)


        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(rnn_hidden_size + img_fc_dim[-1] + action_embedding_size, rnn_hidden_size)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])

        self.num_predefined_action = 1
        self.stop_feat = torch.zeros(batch_size, 1, img_fc_dim[-1], requires_grad=False).to(self.device)
        # self.tensor_mask = torch.zeros(batch_size, max_navigable).to(self.device)

    def forward(self, model_input, input_type):
        """ 
        [Model input]

        img_feat: batch x 36 x img_feature_size
        num_navigable_feat: batch x 3 x 12
        pre_action_feat: previous attended action feature, batch x action_feature_size
        weighted_ctx: batch x rnn_hiddin_size
        navigable_index: list of list
        """
        assert input_type in ['history', 'action']

        if input_type == 'history':
            img_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index = model_input

            batch_size, _, _ = img_feat.size()
            num_navigable_feat = torch.flip(num_navigable_feat, dims=[1])
            num_navigable_attention = num_navigable_feat.view(batch_size, -1)
            num_navigable_attention = F.softmax(num_navigable_attention.float(), dim=1)
            img_feat = img_feat * num_navigable_attention.unsqueeze(-1).expand_as(img_feat)

            # 0~35: navigable location in corresponding heading / 36: stop
            # self.tensor_mask = torch.zeros(batch_size, self.max_navigable).to(self.device)
            navigable_mask = create_new_mask(batch_size, self.max_navigable, navigable_index)  # batch_size x self.max_navigable
            proj_navigable_feat = proj_masking(img_feat, self.proj_navigable_mlp, navigable_mask)
            proj_navigable_feat = torch.cat((proj_navigable_feat, self.stop_feat), 1)
            
            weighted_img_feat, img_attn = self.soft_attn(self.ctx_t_img_fc(weighted_ctx), proj_navigable_feat, mask=navigable_mask)

            # merge info into one LSTM to be carry through time
            concat_input = torch.cat((weighted_ctx, weighted_img_feat, pre_action_feat), 1)

            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))

            return h_1, c_1, weighted_img_feat, img_attn, proj_navigable_feat, navigable_mask
        
        else:  # input_type == 'action'
            proj_navigable_feat, h_1, weighted_ctx = model_input

            h_1_drop = self.dropout(h_1)

            # policy network
            h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
            logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

            return logit

class LowLevelModel(nn.Module):
    def __init__(self, rnn_hidden_size, action_embedding_size, rnn_dropout, fc_bias=True, max_navigable=36, softmax_temperature=30):
        super(LowLevelModel, self).__init__()

        self.max_navigable = max_navigable
        self.hidden_size = rnn_hidden_size
        self.softmax_temperature = 10
        d_num_filters = 4

        # kernel_size, stride, padding, dilation \
        #     = conv_param

        # input_dim, hidden_dim, num_filters, d_kernel_size, input_channel, d_stride, d_padding, d_dilation \
        #     = dynamic_conv2d_param
        
        self.conv1 = nn.Conv2d(in_channels=2*36, out_channels=1*36, kernel_size=5, stride=1,\
                                padding=2, dilation=1, groups=36)
        self.dynamic_conv2d = Dynamic_conv2d(input_dim=rnn_hidden_size, hidden_dim=rnn_hidden_size//4, num_filters=d_num_filters, kernel_size=5, \
                                            input_channel=2, output_channel=1, stride=1, padding=2, dilation=1, groups=36)
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(5, 7), \
                                 dilation=(10, 10))  # (b, 64, 269)
        self.avgpool2_1 = nn.AvgPool2d(kernel_size=3, stride=3)  # (b, 21, 89)
        self.conv2_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(2, 2), \
                                 dilation=(1, 2))  # (b, 10, 43)
        self.avgpool2_2 = nn.AvgPool2d(kernel_size=(3, 9), stride=(3, 3), padding=(0, 0))  # (b, 3, 12)

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(rnn_hidden_size + max_navigable + d_num_filters + action_embedding_size, rnn_hidden_size)

        self.num_predefined_action = 1
        self.logit_fc = nn.Linear(max_navigable, max_navigable + self.num_predefined_action, bias=fc_bias)
        
        self.middle_total_feat_1 = None
        self.middle_total_feat_2 = None
        self.total_visual_feat = None
        # self.tensor_mask = None
    
    def conv2(self, model_input):
        output = self.conv2_1(model_input)
        output = self.avgpool2_1(output)
        output = self.conv2_2(output)
        output = self.avgpool2_2(output)
        output = output.squeeze(1)
        assert output.shape[1] == 3 and output.shape[2] == 12
        
        return output  # (b, 3, 12)
    
    def forward(self, model_input, input_type):
        """ 
        [Model input]

        depth_feat[0](= normalized_raw_depth_feat): batch x 36 x image_h x image_w
        depth_feat[1](= normalized_clip_depth_feat): batch x 36 x image_h x image_w
        obj_detection_feat: batch x 36 x image_h x image_w
        num_navigable_feat: batch x 36
        pre_action_feat(= previous attended action feature): batch x action_feature_size
        weighted_ctx: batch x rnn_hiddin_size
        """
        assert input_type in ['history', 'action']

        if input_type == 'history':
            depth_feat, obj_detection_feat, num_navigable_feat, pre_action_feat, h_0, c_0, weighted_ctx, navigable_index = model_input

            batch_size = depth_feat[0].shape[0]
            image_w = 160
            image_h = 120

            self.total_visual_feat = torch.zeros(batch_size, 1, image_h*3, image_w*12, requires_grad=False).to(device)
            self.middle_total_feat_1 = torch.zeros((batch_size, 2*36, image_h, image_w), requires_grad=False).to(device)
            self.middle_total_feat_2 = torch.zeros((batch_size, 2*36, image_h, image_w), requires_grad=False).to(device)
            # self.tensor_mask = torch.zeros(batch_size, self.max_navigable, requires_grad=False).cuda()

            # navigable_mask = create_new_mask(batch_size, self.max_navigable, navigable_index, self.tensor_mask)
            navigable_mask = create_new_mask(batch_size, self.max_navigable, navigable_index)
            num_navigable_attention = F.softmax(num_navigable_feat.float(), dim=1)  # [b, 36]

            for i in range(self.max_navigable):
                self.middle_total_feat_1[:, 2*i, :, :] = depth_feat[0][:, i, :, :]
                self.middle_total_feat_1[:, 2*i+1, :, :] = depth_feat[1][:, i, :, :]
            total_depth_feat = self.conv1(self.middle_total_feat_1)  # [b, 36, h, w]

            for i in range(self.max_navigable):
                partial_num_navigable_attention = num_navigable_attention[:, i].unsqueeze(-1).unsqueeze(-1).expand(-1, image_h, image_w)
                partial_depth_feat = total_depth_feat[:, i, :, :] * partial_num_navigable_attention # [b, h, w]
                partial_obj_detection_feat = obj_detection_feat[:, i, :, :] * partial_num_navigable_attention # [b, h, w]
                self.middle_total_feat_2[:, 2*i, :, :] = partial_depth_feat
                self.middle_total_feat_2[:, 2*i+1, :, :] = partial_obj_detection_feat
            total_visual_feat, dynamic_filter_attention = self.dynamic_conv2d(weighted_ctx, self.middle_total_feat_2)

            for i in range(self.max_navigable):
                y = i // 12
                x = i % 12
                self.total_visual_feat[:, :, image_h*(2-y):image_h*(3-y), image_w*x:image_w*(x+1)] = total_visual_feat[:, i, :, :].unsqueeze(1)

            visual_feat_1 = self.conv2(self.total_visual_feat)
            visual_feat_2 = self.conv2(torch.cat((self.total_visual_feat[:, :, :, -image_w:], self.total_visual_feat[:, :, :, :-image_w]), dim=-1))
            visual_feat_2 = torch.cat((visual_feat_2[:, :, 1:], visual_feat_2[:, :, 0].unsqueeze(-1)), dim=-1)
            visual_feat = (visual_feat_1 + visual_feat_2)/2

            visual_feat = torch.flip(visual_feat, [1])  # batch x 3 x 12
            visual_feat = visual_feat.view(batch_size, -1)  # batch x 36
            visual_feat = F.softmax(visual_feat/self.softmax_temperature, 1)

            # merge info into one LSTM to be carry through time
            concat_input = torch.cat((weighted_ctx, visual_feat.view(batch_size, -1), dynamic_filter_attention, pre_action_feat), 1)

            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))  #2

            return h_1, c_1, visual_feat, navigable_mask

        else:  # input_type == 'action'
            visual_feat = model_input
            logit = self.logit_fc(visual_feat)

            return logit  # last index correspond to '<STAY>'
