import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
from pytorch_model_summary import summary
from typing import Union
from feature_extraction import *
import torchvision.models as models
import torchaudio
from dataclasses import dataclass
from exp.feature_extraction_exp import *
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


class SSLAASIST(nn.Module):
    def __init__(self):
        super().__init__()

        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.LL = nn.Linear(1024, 128)
        
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)


    def forward(self, x):

        x = x.squeeze(dim=1)

        x = self.LL(x)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden,output


class XLSRAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda', freeze = True, visual=False):
        super(XLSRAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2 = XLSR(
            model_dir=model_dir,
            device=device,
            freeze=freeze,
            visual=visual
        )

        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()
        self.visual = visual
    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
            return last_hidden, output, attention_weights
        # Extract features using XLSRWithPrompt
        features = self.wav2vec2.extract_features(audio_data)

        # Pass the features through W2VAASIST
        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2.eval()   # important

        
class WAVLMAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda', freeze = True):
        super(WAVLMAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wavlm = WAVLM(
            model_dir=model_dir,
            device=device,
            freeze=freeze
        )

        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        features = self.wavlm.extract_features(audio_data)

        # Pass the features through W2VAASIST
        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        self.w2vaasist.eval()
        self.wavlm.eval()   # important       


class MERTAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda',freeze = True):
        super(MERTAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.MERT = MERT(
            model_dir=model_dir,
            device=device,
            freeze=freeze
        )

        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        features = self.MERT.extract_features(audio_data)

        # Pass the features through W2VAASIST
        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.MERT.eval()

        
class ResNet18ForAudio(nn.Module):
    def __init__(self, enc_dim=256, nclasses=2):
        super(ResNet18ForAudio, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.resnet18.fc = nn.Identity()

        self.fc = nn.Linear(512, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=512, power=2, normalized=True)

        self.initialize_params()

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.spec(x.cuda().float()).unsqueeze(dim=1)
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)  
        feat = self.fc(x)
        mu = self.fc_mu(feat)
        return feat, mu
    
class PTW2V2AASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PTW2V2AASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()

class PTW2V2AASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PTW2V2AASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        if self.visual:
            first_hidden, features,  attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return first_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()


class PTWAVLMAASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PTWAVLMAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_WAVLM(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()

class PTMERTAASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PTMERTAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_MERT(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()


            
        

class WPTW2V2AASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens=6, dropout=0.1, visual=False):
        super(WPTW2V2AASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = WPT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens= num_wavelet_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()


class WPTWAVLMAASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens=6, dropout=0.1, visual=False):
        super(WPTWAVLMAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = WPT_WAVLM(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens= num_wavelet_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()


class WPTMERTAASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens=6, dropout=0.1, visual=False):
        super(WPTMERTAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = WPT_MERT(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens= num_wavelet_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()

####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class TypeHead(nn.Module):
    def __init__(self, in_dim=160, n_types=4, hidden_dim=128, dropout=0.1):
        super(TypeHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_types)
        )

    def forward(self, x):
        return self.net(x)


def pool_for_type_head(feats):
    """
    Normalize feature shape for type classifier.
    SSLAASIST returns [B, 160].
    Some other backbones may return [B, T, D].
    """
    if feats.dim() == 3:
        return feats.mean(dim=1)
    if feats.dim() == 2:
        return feats
    return feats.view(feats.size(0), -1)


class TypeRoutedXLSRMERTAASIST(nn.Module):
    """
    Track2 model:
    XLSR expert + MERT expert + sample-adaptive router + AASIST backend.

    XLSR: better for speech/linguistic/prosodic cues.
    MERT: better for singing/music tonal-pitch-rhythm cues.
    """
    def __init__(
        self,
        xlsr_dir,
        mert_dir,
        device='cuda',
        freeze_xlsr=True,
        freeze_mert=True,
        hidden_size=1024,
        router_hidden=256,
        dropout=0.1
    ):
        super(TypeRoutedXLSRMERTAASIST, self).__init__()

        self.freeze_xlsr = freeze_xlsr
        self.freeze_mert = freeze_mert

        self.xlsr = XLSR(
            model_dir=xlsr_dir,
            device=device,
            freeze=freeze_xlsr
        )

        self.mert = MERT(
            model_dir=mert_dir,
            device=device,
            freeze=freeze_mert
        )

        xlsr_dim = getattr(self.xlsr.model.config, "hidden_size", hidden_size)
        mert_dim = getattr(self.mert.model.config, "hidden_size", hidden_size)

        self.proj_xlsr = nn.Linear(xlsr_dim, hidden_size)
        self.proj_mert = nn.Linear(mert_dim, hidden_size)

        # Router input: mean pooled XLSR + mean pooled MERT
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 2)
        )

        # Auxiliary type classifier: speech/sound/singing/music
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 4)
        )

        self.w2vaasist = SSLAASIST()

        self.latest_type_logits = None
        self.latest_expert_weights = None

    def _align_time(self, feat, target_len):
        """
        feat: [B, T, D]
        interpolate T dimension to target_len
        """
        if feat.size(1) == target_len:
            return feat

        feat = feat.transpose(1, 2)  # [B, D, T]
        feat = F.interpolate(
            feat,
            size=target_len,
            mode="linear",
            align_corners=False
        )
        feat = feat.transpose(1, 2)  # [B, T, D]
        return feat

    def forward(self, audio_data):
        x_feat = self.xlsr.extract_features(audio_data)   # [B, Tx, D]
        m_feat = self.mert.extract_features(audio_data)   # [B, Tm, D]

        m_feat = self._align_time(m_feat, x_feat.size(1))

        x_feat = self.proj_xlsr(x_feat)
        m_feat = self.proj_mert(m_feat)

        x_pool = x_feat.mean(dim=1)
        m_pool = m_feat.mean(dim=1)

        router_input = torch.cat([x_pool, m_pool], dim=-1)

        expert_logits = self.router(router_input)
        expert_weights = F.softmax(expert_logits, dim=-1)  # [B, 2]

        self.latest_expert_weights = expert_weights
        self.latest_type_logits = self.type_head(router_input)

        fused = (
            expert_weights[:, 0].view(-1, 1, 1) * x_feat +
            expert_weights[:, 1].view(-1, 1, 1) * m_feat
        )

        last_hidden, output = self.w2vaasist(fused)
        return last_hidden, output

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_xlsr:
            self.xlsr.model.eval()
            for p in self.xlsr.model.parameters():
                p.requires_grad = False

        if self.freeze_mert:
            self.mert.model.eval()
            for p in self.mert.model.parameters():
                p.requires_grad = False

        return self

    def eval(self):
        super().eval()
        return self
####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT


# ============================================================
# UFM-Track2-Full Modules
# ============================================================

class ComplexArtifactEncoder(nn.Module):
    """
    Complex forensic artifact branch.

    Input:
        wav: [B, L]

    Build 4-channel STFT forensic representation:
        1. log magnitude
        2. temporal phase difference
        3. frequency phase difference / group-delay-like cue
        4. local spectral residual

    Output:
        artifact tokens: [B, T, D]
    """
    def __init__(
        self,
        out_dim=512,
        n_fft=512,
        hop_length=160,
        win_length=400,
        dropout=0.1
    ):
        super(ComplexArtifactEncoder, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Dropout(dropout)
        )

        self.proj = nn.Linear(256, out_dim)

    def forward(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        window = torch.hann_window(
            self.win_length,
            device=wav.device,
            dtype=wav.dtype
        )

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )  # [B, F, T]

        mag = torch.abs(stft).clamp(min=1e-6)
        logmag = torch.log(mag)

        phase = torch.angle(stft)

        # Temporal phase difference
        phase_dt = phase[:, :, 1:] - phase[:, :, :-1]
        phase_dt = F.pad(phase_dt, (1, 0))

        # Frequency phase difference, group-delay-like
        phase_df = phase[:, 1:, :] - phase[:, :-1, :]
        phase_df = F.pad(phase_df, (0, 0, 1, 0))

        # Local spectral residual
        smooth = F.avg_pool2d(
            logmag.unsqueeze(1),
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2)
        ).squeeze(1)

        residual = logmag - smooth

        x = torch.stack(
            [logmag, phase_dt, phase_df, residual],
            dim=1
        )  # [B, 4, F, T]

        x = self.cnn(x)      # [B, 256, F', T]
        x = x.mean(dim=2)    # [B, 256, T]
        x = x.transpose(1, 2)  # [B, T, 256]
        x = self.proj(x)     # [B, T, out_dim]

        return x
    
class FullForgeryMemory(nn.Module):
    """
    Shared + type-conditioned forgery memory.

    Memory banks:
        real_shared: common real prototypes
        fake_shared: common fake prototypes
        fake_type: type-conditioned fake prototypes
                   speech/sound/singing/music

    Input:
        tokens: [B, T, D]
        type_prior: [B, 4]

    Output:
        enhanced tokens: [B, T, D]
        memory gap: [B, 2D]
    """
    def __init__(
        self,
        dim=512,
        slots=16,
        n_types=4,
        dropout=0.1
    ):
        super(FullForgeryMemory, self).__init__()

        self.real_shared = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.fake_shared = nn.Parameter(torch.randn(slots, dim) * 0.02)

        self.fake_type = nn.Parameter(
            torch.randn(n_types, slots, dim) * 0.02
        )

        self.norm = nn.LayerNorm(dim)

        self.update = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.res_scale = nn.Parameter(torch.tensor(0.0))

        # Start memory update close to identity to avoid early instability.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def attend(self, tokens, memory):
        """
        tokens: [B, T, D]
        memory:
            [K, D] or [B, K, D]
        """
        scale = tokens.size(-1) ** -0.5

        if memory.dim() == 2:
            memory = memory.unsqueeze(0).expand(tokens.size(0), -1, -1)

        attn = torch.softmax(
            torch.matmul(tokens, memory.transpose(-1, -2)) * scale,
            dim=-1
        )

        ctx = torch.matmul(attn, memory)
        return ctx

    def forward(self, tokens, type_prior=None):
        z = self.norm(tokens)

        B = z.size(0)
        n_types = self.fake_type.size(0)

        if type_prior is None:
            type_prior = torch.ones(
                B,
                n_types,
                device=z.device,
                dtype=z.dtype
            ) / n_types

        real_ctx = self.attend(z, self.real_shared)
        fake_shared_ctx = self.attend(z, self.fake_shared)

        # Mixture of type-conditioned fake memories
        type_mem = torch.einsum(
            "bn,nkd->bkd",
            type_prior,
            self.fake_type
        )

        fake_type_ctx = self.attend(z, type_mem)

        gap_shared = fake_shared_ctx - real_ctx
        gap_type = fake_type_ctx - real_ctx

        delta = self.update(
            torch.cat([z, gap_shared, gap_type], dim=-1)
        )

        if not torch.isfinite(delta).all():
            with torch.no_grad():
                y = torch.nan_to_num(delta.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"[NONFINITE MEMORY] delta | "
                    f"shape={tuple(delta.shape)} | "
                    f"nan={torch.isnan(delta).any().item()} | "
                    f"inf={torch.isinf(delta).any().item()} | "
                    f"min={y.min().item():.4e} | "
                    f"max={y.max().item():.4e}"
                )
            raise RuntimeError("Non-finite memory delta")

        delta = torch.clamp(delta, min=-10.0, max=10.0)

        enhanced = tokens + torch.tanh(self.res_scale) * delta

        pooled_gap = torch.cat(
            [
                gap_shared.mean(dim=1),
                gap_type.mean(dim=1)
            ],
            dim=-1
        )  # [B, 2D]

        return enhanced, pooled_gap
    
class FullExpertRouter(nn.Module):
    """
    Uncertainty-aware expert router.

    Experts:
        0 XLSR
        1 MERT
        2 BEATs/OpenBEATs
        3 Artifact/Memory stream
    """
    def __init__(
        self,
        dim=512,
        n_experts=4,
        n_types=4,
        dropout=0.1
    ):
        super(FullExpertRouter, self).__init__()

        in_dim = dim * 7

        self.router = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_experts)
        )

        self.type_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_types)
        )

    def forward(
        self,
        x_pool,
        m_pool,
        b_pool,
        a_pool,
        f_pool,
        mem_gap
    ):
        """
        x_pool: [B, D]
        m_pool: [B, D]
        b_pool: [B, D]
        a_pool: [B, D]
        f_pool: [B, D]
        mem_gap: [B, 2D]
        """
        h = torch.cat(
            [x_pool, m_pool, b_pool, a_pool, f_pool, mem_gap],
            dim=-1
        )  # [B, 7D]

        expert_logits = self.router(h)
        type_logits = self.type_head(h)

        expert_weights = torch.softmax(expert_logits, dim=-1)
        type_prob = torch.softmax(type_logits, dim=-1)

        return expert_weights, type_logits, type_prob
    
class CrossStreamFusionBlock(nn.Module):
    """
    Bidirectional semantic-forensic cross-attention.
    """
    def __init__(
        self,
        dim=512,
        num_heads=8,
        dropout=0.1
    ):
        super(CrossStreamFusionBlock, self).__init__()

        self.sem_norm = nn.LayerNorm(dim)
        self.for_norm = nn.LayerNorm(dim)
        
        self.attn_scale = nn.Parameter(torch.tensor(0.0))
        self.ffn_scale = nn.Parameter(torch.tensor(0.0))

        self.sem_to_for = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.for_to_sem = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.sem_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        self.for_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def _check_finite(self, name, x):
        if not torch.isfinite(x).all():
            with torch.no_grad():
                y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"[NONFINITE CROSS] {name} | "
                    f"shape={tuple(x.shape)} | "
                    f"nan={torch.isnan(x).any().item()} | "
                    f"inf={torch.isinf(x).any().item()} | "
                    f"min={y.min().item():.4e} | "
                    f"max={y.max().item():.4e}"
                )
            raise RuntimeError(f"Non-finite cross tensor: {name}")
        return x

    def forward(self, semantic_tokens, forensic_tokens):
        sem = self.sem_norm(semantic_tokens)
        forg = self.for_norm(forensic_tokens)

        sem_ctx, _ = self.sem_to_for(
            query=sem,
            key=forg,
            value=forg,
            need_weights=False
        )

        for_ctx, _ = self.for_to_sem(
            query=forg,
            key=sem,
            value=sem,
            need_weights=False
        )

        sem_ctx = self._check_finite("sem_ctx", sem_ctx)
        for_ctx = self._check_finite("for_ctx", for_ctx)

        sem_ctx = torch.clamp(sem_ctx, -10.0, 10.0)
        for_ctx = torch.clamp(for_ctx, -10.0, 10.0)

        sem_ctx = torch.clamp(sem_ctx, min=-10.0, max=10.0)
        for_ctx = torch.clamp(for_ctx, min=-10.0, max=10.0)

        semantic_tokens = semantic_tokens + torch.tanh(self.attn_scale) * sem_ctx
        forensic_tokens = forensic_tokens + torch.tanh(self.attn_scale) * for_ctx

        sem_ffn = self.sem_ffn(semantic_tokens)
        for_ffn = self.for_ffn(forensic_tokens)

        sem_ffn = self._check_finite("sem_ffn", sem_ffn)
        for_ffn = self._check_finite("for_ffn", for_ffn)

        sem_ffn = torch.clamp(sem_ffn, -10.0, 10.0)
        for_ffn = torch.clamp(for_ffn, -10.0, 10.0)

        sem_ffn = torch.clamp(sem_ffn, min=-10.0, max=10.0)
        for_ffn = torch.clamp(for_ffn, min=-10.0, max=10.0)

        semantic_tokens = semantic_tokens + torch.tanh(self.ffn_scale) * sem_ffn
        forensic_tokens = forensic_tokens + torch.tanh(self.ffn_scale) * for_ffn
        
        return semantic_tokens, forensic_tokens


class BiCrossStreamTransformer(nn.Module):
    def __init__(
        self,
        dim=512,
        heads=8,
        layers=2,
        dropout=0.1
    ):
        super(BiCrossStreamTransformer, self).__init__()

        self.layers = nn.ModuleList([
            CrossStreamFusionBlock(
                dim=dim,
                num_heads=heads,
                dropout=dropout
            )
            for _ in range(layers)
        ])

    def forward(self, sem, forg):
        for layer in self.layers:
            sem, forg = layer(sem, forg)

        return sem, forg
    
class ForgeryQueryDecoder(nn.Module):
    """
    Learnable real/fake query decoder.
    """
    def __init__(
        self,
        dim=1024,
        heads=8,
        dropout=0.1
    ):
        super(ForgeryQueryDecoder, self).__init__()

        self.queries = nn.Parameter(torch.randn(2, dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

        self.cls = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )

    def forward(self, tokens):
        B = tokens.size(0)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        q_out, _ = self.attn(
            query=q,
            key=tokens,
            value=tokens,
            need_weights=False
        )

        q_out = self.norm(q_out)

        real_q = q_out[:, 0]
        fake_q = q_out[:, 1]

        logits = self.cls(
            torch.cat([real_q, fake_q], dim=-1)
        )

        return logits
    
class UFMTrack2Full(nn.Module):
    """
    UFM-Track2-Full:
        XLSR Expert
        MERT Expert
        BEATs/OpenBEATs Expert
        Complex Artifact Encoder
        Shared + Type-conditioned Forgery Memory
        Uncertainty-aware Expert Router
        Semantic-Forensic Cross-stream Transformer
        Forgery Query Decoder
    """
    def __init__(
        self,
        xlsr_dir,
        mert_dir,
        beats_dir,
        device="cuda",
        freeze_xlsr=True,
        freeze_mert=True,
        freeze_beats=True,
        dim=512,
        mem_slots=16,
        heads=8,
        layers=2,
        dropout=0.1
    ):
        super(UFMTrack2Full, self).__init__()

        self.freeze_xlsr = freeze_xlsr
        self.freeze_mert = freeze_mert
        self.freeze_beats = freeze_beats

        self.xlsr = XLSR(
            model_dir=xlsr_dir,
            device=device,
            freeze=freeze_xlsr
        )

        self.mert = MERT(
            model_dir=mert_dir,
            device=device,
            freeze=freeze_mert
        )

        self.beats = OpenBEATS(
            model_dir=beats_dir,
            device=device,
            freeze=freeze_beats
        )

        xlsr_dim = getattr(self.xlsr.model.config, "hidden_size", 1024)
        mert_dim = getattr(self.mert.model.config, "hidden_size", 1024)
        self.proj_xlsr = nn.Linear(xlsr_dim, dim)
        self.proj_mert = nn.Linear(mert_dim, dim)
        self.proj_beats = nn.Linear(1024, dim)

        self.norm_xlsr = nn.LayerNorm(dim)
        self.norm_mert = nn.LayerNorm(dim)
        self.norm_beats = nn.LayerNorm(dim)
        self.norm_artifact = nn.LayerNorm(dim)
        self.norm_memory = nn.LayerNorm(dim)
        self.norm_fused = nn.LayerNorm(dim * 2)

        self.artifact = ComplexArtifactEncoder(
            out_dim=dim,
            dropout=dropout
        )

        self.memory = FullForgeryMemory(
            dim=dim,
            slots=mem_slots,
            n_types=4,
            dropout=dropout
        )

        self.router = FullExpertRouter(
            dim=dim,
            n_experts=4,
            n_types=4,
            dropout=dropout
        )

        self.cross = BiCrossStreamTransformer(
            dim=dim,
            heads=heads,
            layers=layers,
            dropout=dropout
        )

        self.decoder = ForgeryQueryDecoder(
            dim=dim * 2,
            heads=heads,
            dropout=dropout
        )

        self.latest_expert_weights = None
        self.latest_type_logits = None
    
    def _safe(self, x, clamp=20.0):
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=clamp,
            neginf=-clamp
        )
        x = torch.clamp(x, min=-clamp, max=clamp)
        return x
    
    def _safe_frozen(self, x, clamp=20.0):
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=clamp,
            neginf=-clamp
        )
        x = torch.clamp(x, min=-clamp, max=clamp)
        return x


    def _check_finite(self, name, x):
        if not torch.isfinite(x).all():
            with torch.no_grad():
                y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                print(
                    f"[NONFINITE FORWARD] {name} | "
                    f"shape={tuple(x.shape)} | "
                    f"nan={torch.isnan(x).any().item()} | "
                    f"inf={torch.isinf(x).any().item()} | "
                    f"min={y.min().item():.4e} | "
                    f"max={y.max().item():.4e} | "
                    f"mean={y.mean().item():.4e} | "
                    f"std={y.std().item():.4e}"
                )
            raise RuntimeError(f"Non-finite forward tensor: {name}")
        return x
    
    def _align_time(self, feat, target_len):
        if feat.size(1) == target_len:
            return feat

        feat = feat.transpose(1, 2)

        feat = F.interpolate(
            feat,
            size=target_len,
            mode="linear",
            align_corners=False
        )

        feat = feat.transpose(1, 2)

        return feat

    def forward(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        # =====================================================
        # 1. Multi-domain experts
        # =====================================================
        xlsr_feat = self._safe_frozen(self.xlsr.extract_features(wav).float())
        mert_feat = self._safe_frozen(self.mert.extract_features(wav).float())
        beats_feat = self._safe_frozen(self.beats.extract_features(wav).float())

        x = self.proj_xlsr(xlsr_feat)
        m = self.proj_mert(mert_feat)
        b = self.proj_beats(beats_feat)

        x = self._check_finite("proj_xlsr", x)
        m = self._check_finite("proj_mert", m)
        b = self._check_finite("proj_beats", b)

        x = self.norm_xlsr(torch.clamp(x, -20.0, 20.0))
        m = self.norm_mert(torch.clamp(m, -20.0, 20.0))
        b = self.norm_beats(torch.clamp(b, -20.0, 20.0))
        T = x.size(1)

        m = self._align_time(m, T)
        b = self._align_time(b, T)

        # =====================================================
        # 2. Complex artifact branch
        # =====================================================
        a = self.artifact(wav)
        a = self._align_time(a, T)
        a = self._check_finite("artifact", a)
        a = self.norm_artifact(torch.clamp(a, -20.0, 20.0))

        # =====================================================
        # 3. First-pass memory with uniform type prior
        # =====================================================
        B = wav.size(0)

        uniform_type_prior = torch.ones(
            B,
            4,
            device=wav.device,
            dtype=x.dtype
        ) / 4.0

        a_mem, mem_gap = self.memory(
            a,
            type_prior=uniform_type_prior
        )
        a_mem = self._check_finite("a_mem_first", a_mem)
        mem_gap = self._check_finite("mem_gap_first", mem_gap)

        a_mem = self.norm_memory(torch.clamp(a_mem, -20.0, 20.0))
        mem_gap = torch.clamp(mem_gap, -20.0, 20.0)
        
        # =====================================================
        # 4. Router
        # =====================================================
        x_pool = x.mean(dim=1)
        m_pool = m.mean(dim=1)
        b_pool = b.mean(dim=1)
        a_pool = a.mean(dim=1)
        f_pool = a_mem.mean(dim=1)

        expert_w, type_logits, type_prob = self.router(
            x_pool,
            m_pool,
            b_pool,
            a_pool,
            f_pool,
            mem_gap
        )
        expert_w = self._check_finite("expert_w", expert_w)
        type_logits = self._check_finite("type_logits", type_logits)
        type_prob = self._check_finite("type_prob", type_prob)

        # =====================================================
        # 5. Second-pass memory with predicted type prior
        # =====================================================
        a_mem, mem_gap = self.memory(
            a,
            type_prior=type_prob
        )
        a_mem = self._check_finite("a_mem_second", a_mem)
        mem_gap = self._check_finite("mem_gap_second", mem_gap)

        a_mem = self.norm_memory(torch.clamp(a_mem, -20.0, 20.0))
        mem_gap = torch.clamp(mem_gap, -20.0, 20.0)
        
        self.latest_expert_weights = expert_w
        self.latest_type_logits = type_logits

        # =====================================================
        # 6. Semantic expert fusion
        # =====================================================
        sem_w = expert_w[:, :3]
        sem_w = sem_w / (sem_w.sum(dim=-1, keepdim=True) + 1e-8)

        semantic = (
            sem_w[:, 0].view(-1, 1, 1) * x +
            sem_w[:, 1].view(-1, 1, 1) * m +
            sem_w[:, 2].view(-1, 1, 1) * b
        )

        # =====================================================
        # 7. Forensic fusion
        # =====================================================
        art_gate = expert_w[:, 3].view(-1, 1, 1)

        forensic = (
            art_gate * a_mem +
            (1.0 - art_gate) * a
        )

        # =====================================================
        # 8. Cross-stream interaction
        # =====================================================
        semantic, forensic = self.cross(
            semantic,
            forensic
        )

        semantic = self._check_finite("semantic_after_cross", semantic)
        forensic = self._check_finite("forensic_after_cross", forensic)

        semantic = torch.clamp(semantic, -20.0, 20.0)
        forensic = torch.clamp(forensic, -20.0, 20.0)

        fused = torch.cat(
            [semantic, forensic],
            dim=-1
        )

        fused = self._check_finite("fused", fused)
        fused = self.norm_fused(torch.clamp(fused, -20.0, 20.0))
        
        # =====================================================
        # 9. Forgery query decoder
        # =====================================================
        logits = self.decoder(fused)
        logits = self._check_finite("raw_logits", logits)
        logits = torch.clamp(logits, -30.0, 30.0)

        pooled = fused.mean(dim=1)

        return pooled, logits

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_xlsr:
            self.xlsr.model.eval()
            for p in self.xlsr.model.parameters():
                p.requires_grad = False

        if self.freeze_mert:
            self.mert.model.eval()
            for p in self.mert.model.parameters():
                p.requires_grad = False

        if self.freeze_beats:
            self.beats.model.eval()
            for p in self.beats.model.parameters():
                p.requires_grad = False

        return self

    def eval(self):
        super().eval()
        return self
        
if __name__ == "__main__":
    print("model.py loaded successfully.")
