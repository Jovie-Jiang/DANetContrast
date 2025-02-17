from torch import nn, einsum
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch.nn.functional as F
import cv2

"""
 PreNorm 类，将一个包含 Layer Normalization 的前馈模块封装在一起
 其中 dim 是输入向量的维度, fn 是一个前馈函数
 在前馈函数的计算之前，先对输入进行归一化处理，以使得模型更加稳定和高效
"""
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

"""
Factorized Self-Attention 类，实现了一个自注意力机制，将输入向量分为多个头进行处理
"""
class FSAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

"""
Factorized Dot-product Attention 类，实现了一个因子化的点积注意力机制，将输入向量分为多个头进行处理
"""
class FDAttention(nn.Module):
    def __init__(self, dim, nt, nh, nw, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.nt = nt
        self.nh = nh
        self.nw = nw

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        qs, qt = q.chunk(2, dim=1)
        ks, kt = k.chunk(2, dim=1)
        vs, vt = v.chunk(2, dim=1)

        # Attention over spatial dimension
        qs = qs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        ks, vs = ks.view(b, h // 2, self.nt, self.nh * self.nw, -1), vs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        spatial_dots = einsum('b h t i d, b h t j d -> b h t i j', qs, ks) * self.scale
        sp_attn = self.attend(spatial_dots)
        spatial_out = einsum('b h t i j, b h t j d -> b h t i d', sp_attn, vs)

        # Attention over temporal dimension
        qt = qt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        kt, vt = kt.view(b, h // 2, self.nh * self.nw, self.nt, -1), vt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        temporal_dots = einsum('b h s i d, b h s j d -> b h s i j', qt, kt) * self.scale
        temporal_attn = self.attend(temporal_dots)
        temporal_out = einsum('b h s i j, b h s j d -> b h s i d', temporal_attn, vt)

        # return self.to_out(out)

"""
一个前馈神经网络类，包含了一个线性层和一个 GELU 激活函数，用于处理输入向量
"""
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

"""
Factorized Self-Attention Transformer 编码器类，将输入向量进行多层自注意力机制和前馈网络处理
"""
class FSATransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, nt, nh, nw, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))

    def forward(self, x):

        b = x.shape[0]
        x = torch.flatten(x, start_dim=0, end_dim=1)  # extract spatial tokens from x

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = sp_attn_x.chunk(b, dim=0)
            sp_attn_x = [temp[None] for temp in sp_attn_x]
            sp_attn_x = torch.cat(sp_attn_x, dim=0).transpose(1, 2)
            sp_attn_x = torch.flatten(sp_attn_x, start_dim=0, end_dim=1)

            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention

            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = x.chunk(b, dim=0)
            x = [temp[None] for temp in x]
            x = torch.cat(x, dim=0).transpose(1, 2)
            x = torch.flatten(x, start_dim=0, end_dim=1)

        # Reshape vector to [b, nt*nh*nw, dim]
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        return x

"""
Factorized Dot-product Attention Transformer 编码器类，它将输入向量进行多层因子化点积注意力机制处理
"""
class FDATransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, nt, nh, nw, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, FDAttention(dim, nt, nh, nw, heads=heads, dim_head=dim_head, dropout=dropout)))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x

        return x

"""
 ViViT 模型的主体类，它将输入的视频数据映射成一个特征向量，并通过 Transformer 编码器处理后得到最终的分类结果
"""
class ViViTBackbone(nn.Module):
    def __init__(self, t, h, w, patch_t, patch_h, patch_w, num_classes, dim, depth, heads, mlp_dim, dim_head=3,
                 channels=2, mode='tubelet', device='cuda', emb_dropout=0., dropout=0., model=3):
        super().__init__()

        assert t % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0, "Video dimensions should be divisible by " \
                                                                           "tubelet size "

        self.T = t
        self.H = h
        self.W = w
        self.channels = channels
        self.t = patch_t
        self.h = patch_h
        self.w = patch_w
        self.mode = mode
        self.device = device

        self.nt = self.T // self.t
        self.nh = self.H // self.h
        self.nw = self.W // self.w

        tubelet_dim = self.t * self.h * self.w * channels
        # print(tubelet_dim)
        # 288

        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.t, ph=self.h, pw=self.w),
            nn.Linear(tubelet_dim, dim)
        )

        # repeat same spatial position encoding temporally
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to(device)


        self.dropout = nn.Dropout(emb_dropout)

        if model == 3:
            self.transformer = FSATransformerEncoder(dim, depth, heads, dim_head, mlp_dim,
                                                     self.nt, self.nh, self.nw, dropout)
        elif model == 4:
            assert heads % 2 == 0, "Number of heads should be even"
            self.transformer = FDATransformerEncoder(dim, depth, heads, dim_head, mlp_dim,
                                                     self.nt, self.nh, self.nw, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, input_h, input_v, drop_h, drop_v):

        raw_h = torch.cat((input_h, drop_h), dim=1)  # 合并 input_h 和 dop_h
        raw_v = torch.cat((input_v, drop_v), dim=1)  # 合并 input_v 和 dop_v

        # print(raw_h.shape) 
        # raw_h1 = torch.reshape(raw_h, (64, 64))
        # raw_v1 = torch.reshape(raw_v, (64, 64))
        # drop_h1 = torch.reshape(drop_h, (64, 64))
        # drop_v1 = torch.reshape(drop_v, (64, 64))

        tokens_raw_h = self.to_tubelet_embedding(raw_h)
        tokens_raw_h += self.pos_embedding
        tokens_raw_h = self.dropout(tokens_raw_h)
        x_raw_h = self.transformer(tokens_raw_h)
        x_raw_h = x_raw_h.mean(dim=1)
        x_raw_h = self.to_latent(x_raw_h)

        tokens_raw_v = self.to_tubelet_embedding(raw_v)
        tokens_raw_v += self.pos_embedding
        tokens_raw_v = self.dropout(tokens_raw_v)
        x_raw_v = self.transformer(tokens_raw_v)
        x_raw_v = x_raw_v.mean(dim=1)
        x_raw_v = self.to_latent(x_raw_v)

        # tokens_drop_h = self.to_tubelet_embedding(drop_h)
        # tokens_drop_h += self.pos_embedding
        # tokens_drop_h = self.dropout(tokens_drop_h)
        # x_drop_h = self.transformer(tokens_drop_h)
        # x_drop_h = x_drop_h.mean(dim=1)
        # x_drop_h = self.to_latent(x_drop_h)

        # tokens_drop_v = self.to_tubelet_embedding(drop_v)
        # tokens_drop_v += self.pos_embedding
        # tokens_drop_v = self.dropout(tokens_drop_v)
        # x_drop_v = self.transformer(tokens_drop_v)
        # x_drop_v = x_drop_v.mean(dim=1)
        # x_drop_v = self.to_latent(x_drop_v)

        # out = x_raw_h + x_raw_v + x_drop_h + x_drop_v
        
        # logits = self.mlp_head(out)
        # probabilities = F.softmax(logits, dim=1)
        # _, predicted_classes = probabilities.max(dim=1)
        # return predicted_classes
        # # return self.mlp_head(out) 
        out = x_raw_h + x_raw_v
        logits = self.mlp_head(out)
        return logits
    """
    def forward(self, x):
        #  x is a video: (b, C, T, H, W) 

        tokens = self.to_tubelet_embedding(x)
        tokens += self.pos_embedding
        tokens = self.dropout(tokens)

        x = self.transformer(tokens)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)
    """

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # raw_h = torch.rand(64, 1, 60, 30, 110).to(device).detach().cpu().numpy()
    # raw_v = torch.rand(64, 1, 60, 30, 110).to(device).detach().cpu().numpy()
    # drop_h = torch.rand(64, 1, 60, 30, 110).to(device).detach().cpu().numpy()
    # drop_v = torch.rand(64, 1, 60, 30, 110).to(device).detach().cpu().numpy()

    raw_h = torch.rand(1, 1, 60, 64, 64).to(device)
    raw_v = torch.rand(1, 1, 60, 64, 64).to(device)
    drop_h = torch.rand(1, 1, 60, 64, 64).to(device)
    drop_v = torch.rand(1, 1, 60, 64, 64).to(device)


    vivit = ViViTBackbone(
            t=60,h=64, w=64,  # 输入视频的时间维度（长度）、高度和宽度
            patch_t=6, patch_h=4, patch_w=4,  #每个时间维度、高度和宽度上的图像块大小
            num_classes=9,  # 分类的类别数
            dim=512,        # 模型中特征的维度
            depth=6,        # 模型中的层数
            heads=4,       # 自注意力机制中的头数
            mlp_dim=8,      # MLP（多层感知机）中的隐藏层维度
            model=3         # 模型的类型
        ).to(device)
    
    out = vivit(raw_h, raw_v, drop_h, drop_v)
    print(out)

    # device = torch.device('cpu')
    # x = torch.rand(32, 3, 32, 64, 64).to(device)

    # vivit = ViViTBackbone(32, 64, 64, 8, 4, 4, 10, 512, 6, 10, 8, model=3).to(device)
    # out = vivit(x)
    # print(out)
