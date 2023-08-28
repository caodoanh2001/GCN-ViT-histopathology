import torch.nn.functional as F
import torch.utils.data as data
import torch
import timm
from gcn import GCNBlock
import torch.nn as nn
import numpy as np
from skimage.util.shape import view_as_windows
from att_utils import MultiheadAttention

class CrossAttention(nn.Module):
    '''
    Cross-Attention between two branches. Originaly introduced in https://github.com/IBM/CrossViT
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MHSA(nn.Module):
    '''
    define a MHSA class for stacking in nn.ModuleList()
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mhsa = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

    def forward(self, x, att_mask=None):
        return self.mhsa(x, x, x, attn_mask=att_mask)[0]

class GCNViTClsNodeNewRelationshipBestModel(nn.Module):
    '''
    # //
    # //                       _oo0oo_
    # //                      o8888888o
    # //                      88" . "88
    # //                      (| -_- |)
    # //                      0\  =  /0
    # //                    ___/`---'\___
    # //                  .' \\|     |// '.
    # //                 / \\|||  :  |||// \
    # //                / _||||| -:- |||||- \
    # //               |   | \\\  -  /// |   |
    # //               | \_|  ''\---/''  |_/ |
    # //               \  .-\__  '-'  ___/-. /
    # //             ___'. .'  /--.--\  `. .'___
    # //          ."" '<  `.___\_<|>_/___.' >' "".
    # //         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
    # //         \  \ `_.   \_ __\ /__ _/   .-` /  /
    # //     =====`-.____`.___ \_____/___.-`___.-'=====
    # //                       `=---='
    # //
    # //
    # //     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    iter 800

    C-Test I
    Precision:  0.8895073404368559
    Recall:  0.8186716913216316
    F1:  0.8473840389320774
    Accuracy:  0.8897984886649875
    Kw: 0.9463112641423352
    Confusion matrix: 
    [[450   2   1   0]
    [  1 119  72   0]
    [  0  30 697  11]
    [  0   0  58 147]]
    
    C-Test II
    Precision:  0.7335359595826049
    Recall:  0.8079372000417291
    F1:  0.7589119640566256
    Accuracy:  0.831950621766361
    Kw: 0.8948576075160781
    Confusion matrix:
    [[26529   876   187   304]
    [   47  4972  2925   450]
    [   90  6931 49587  5377]
    [    4    30  1293 10568]]

    P-Test I
    Precision:  0.7392140938206748
    Recall:  0.5978582983451077
    F1:  0.640126471548184
    Accuracy:  0.7196129336794902
    Kw: 0.6272169921351793
    Confusion matrix: 
    [[  64   61    2    0]
    [  10 1304  285    3]
    [   6  531 1548   36]
    [   5   27  222  133]]

    P-Test II
    Precision:  0.7052750501269298
    Recall:  0.714412515199037
    F1:  0.694699418790589
    Accuracy:  0.7994843548576116
    Kw: 0.7318810137896015
    Confusion matrix: 
    [[ 733  519   30    2]
    [  75 5035  740    2]
    [  80 1650 7720  232]
    [   1    5   86  156]]
    '''
    def __init__(self, num_nodes=None, node_dim=None, embed_dim=None):
        super().__init__()

        self.num_gcns = 12
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.num_classes = 4

        # GCNs
        self.cls_node_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_node_token, std=1e-6)
        
        self.gcn_class = nn.ModuleList([
            GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            for _ in range(self.num_gcns)
        ])
        
        self.gcn_nodes = nn.ModuleList([
            GCNBlock(self.node_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            if (i == 0) else \
            GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            for i in range(self.num_gcns)
        ])

        self.cls_mhsa_classes = nn.ModuleList(
            [
                MHSA(embed_dim=self.embed_dim, num_heads=8)
                for _ in range(self.num_gcns)
            ]
        )

        # ViT
        vit = timm.create_model('vit_base_patch32_384', num_classes=0, pretrained=True).cuda()
        self.vit_pre = vit.patch_embed
        self._pos_embed = vit._pos_embed
        self.vit_blocks = nn.ModuleList([*vit.blocks]).cuda()
        
        # ViT & Graph relationships
        vit_regions = view_as_windows(np.arange(0, 144).reshape(12, 12), (3, 3), step=3).reshape(-1, 9)
        self.g2v_relationships = torch.from_numpy(vit_regions).cuda()
        self.cross_atts = nn.ModuleList([
            CrossAttention(dim=self.embed_dim, num_heads=8)
            for _ in range(self.num_gcns)
        ])

        self.cross_att_gcn_vit = CrossAttention(dim=self.embed_dim, num_heads=8)
        self.cross_att_vit_gcn = CrossAttention(dim=self.embed_dim, num_heads=8)

        # Classifier
        self.classifier_ = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 1, self.num_classes),
        )

    def get_vit_tokens_by_indices(self, tokens, indices):
        patch_tokens = []
        for ind in indices:
            patch_tokens.append(tokens[:, ind, :].unsqueeze(1))
        return patch_tokens

    def forward(self, imgs, graphs, adjs):
        imgs = imgs.permute(0, 3, 1, 2)
        bs, _, _ = graphs.shape

        # ViT branch
        vit_tokens = self.vit_pre(imgs)
        vit_tokens = self._pos_embed(vit_tokens)

        # adjs for patch nodes
        nodes_adjs = adjs[:, 1:, 1:]
        masks = torch.ones(graphs.shape[:2]).to(adjs)

        ## STEP 1
        # Patch nodes update
        for node_layer in self.gcn_nodes:
            graphs = node_layer(graphs, nodes_adjs, masks)

        # Concatnate [cls] node
        cls_token = self.cls_node_token.repeat(bs, 1, 1)
        graphs = torch.cat([cls_token, graphs], dim=1)
        masks = torch.ones(graphs.shape[:2]).to(adjs)

        ## STEP 2
        # [cls] node update and ViT-GCN interactions
        for _, (cls_layer, vit_block, cls_mhsa_layer, cross_att) in enumerate(zip(self.gcn_class, self.vit_blocks, self.cls_mhsa_classes, self.cross_atts)):
            # ViT branch
            vit_tokens = vit_block(vit_tokens)

            # Update [cls] node along with node tokens
            graphs = cls_layer(graphs, adjs, masks)
            graphs += cls_mhsa_layer(graphs, att_mask=(adjs == 0).repeat(8, 1, 1))

            # Graph-ViT interactions
            patch_vit_tokens = self.get_vit_tokens_by_indices(vit_tokens[:, 1:, :], self.g2v_relationships)
            patch_vit_tokens = torch.cat(patch_vit_tokens, dim=1) # bs, 16, 9, 768
            patch_vit_tokens = patch_vit_tokens.view(-1, self.g2v_relationships.shape[-1], self.embed_dim) # bs * 16, 9, 768

            reshaped_graphs = graphs[:, 1:].reshape(-1, self.embed_dim).unsqueeze(1) # bs * 16, 1, 768
            relationship = reshaped_graphs + cross_att(torch.cat([reshaped_graphs, patch_vit_tokens], dim=1))
            graphs[:, 1:] = relationship.reshape(bs, self.g2v_relationships.shape[0], self.embed_dim) # bs, 1, 768

        ## STEP 3: CrossViT
        graphs[:, 0:1] += self.cross_att_gcn_vit(torch.cat([graphs[:, 0:1, :], vit_tokens[:, 1:]], dim=1))
        vit_tokens[:, 0:1] += self.cross_att_vit_gcn(torch.cat([vit_tokens[:, 0:1, :], graphs[:, 1:]], dim=1))

        # Classification using only [cls] node
        ce_logits = [vit_tokens[:, 0], graphs[:, 0]]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        probs = self.classifier_(ce_logits)
        return probs
    
select_model = GCNViTClsNodeNewRelationshipBestModel