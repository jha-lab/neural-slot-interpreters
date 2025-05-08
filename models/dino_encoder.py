import argparse
from collections import namedtuple
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('../')
from models.slot_attention import SlotAttentionEncoder
from models.model_utils import dino_vit_base_patch8_224


class BroadCastDecoder(nn.Module):


    def __init__(self,hidden_dim,slot_dim):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.slot_dim = slot_dim
        self.mlp  = nn.Sequential(
            nn.Linear(slot_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim+1)
        )
        self.postional_encoding = nn.Embedding(784,slot_dim)
    
    def forward(self, slots):
        B, K, _ = slots.shape
        slots = slots.view(B,K,1,self.slot_dim).repeat(1,1,784,1)
        pos = torch.arange(784).to(slots.device)
        pos = self.postional_encoding(pos)
        slots = slots + pos
        slots = self.mlp(slots) # (B, K, 784, hidden_dim+1)
        alpha_masks = slots[..., 0] # (B, K, 784)
        alpha_masks = F.softmax(alpha_masks, dim=1) # (B, K, 784)
        features_slots = slots[..., 1:] # (B, K, 784, hidden_dim)
        features = (features_slots * alpha_masks.unsqueeze(-1)).sum(dim=1) # (B, 784, hidden_dim)
        alpha_masks = alpha_masks.view(B, K, 28, 28)
        return features, alpha_masks 
        


class ImageEncoder(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.slot_dim = args.slot_dim
        self.encoder = dino_vit_base_patch8_224(pretrained=True)
        self.slot_attn = SlotAttentionEncoder(
            num_iterations=args.num_iterations,
            num_slots=args.num_slots,
            num_heads=args.num_slot_heads,
            input_channels=768,
            slot_size=args.slot_dim,
            mlp_hidden_size=args.mlp_hidden_size
        )
        self.H_enc = args.image_size // 8
        self.W_enc = args.image_size // 8
        self.H = args.image_size
        self.W = args.image_size
        self.num_slots = args.num_slots
        self.embed_proj =  nn.Linear(args.slot_dim, args.embed_dim)
        self.norm_embed = nn.LayerNorm(args.embed_dim)
        self.embed_mlp = nn.Sequential(
            nn.Linear(args.embed_dim, 4*args.embed_dim),
            nn.ReLU(),
            nn.Linear(4*args.embed_dim, args.embed_dim),
            nn.Dropout(args.dropout)
        )
        self.decoder = BroadCastDecoder(768, args.slot_dim)
        

    def forward(self, x):
        
        B, _, _, _ = x.shape
        with torch.no_grad():
            flat_feature_grid = self.encoder.forward_features(x)
        flat_feature_grid = flat_feature_grid.detach()
        slots, _ = self.slot_attn(flat_feature_grid[:,1:])
        features, attns = self.decoder(slots)
        attns_raw = attns.reshape(-1, self.num_slots, 1, self.H_enc, self.W_enc)
        attns_raw_scale = attns_raw.repeat_interleave(self.H // self.H_enc, dim=-2).repeat_interleave(self.W // self.W_enc, dim=-1)
        attns_vis = x.unsqueeze(1)*attns_raw_scale + (1. - attns_raw_scale)
        
        #proj slots
        slots_proj = self.embed_proj(slots)
        slots_proj = slots_proj + self.embed_mlp(self.norm_embed(slots_proj))

        mse = ((features - flat_feature_grid[:,1:])**2).sum(-1).mean()
        return slots_proj, attns_vis, attns_raw, slots, mse



def main():

    # unit tests
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--slot_dim', type=int, default=192)
    parser.add_argument('--num_slot_heads', type=int, default=1)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--obs_temp', type=float, default=10)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--img_channels', type=int, default=3)

    args = parser.parse_args()

    model = ImageEncoder(args)
    model = model.cuda()
    images = torch.randn(2,3,224,224).cuda()

    proj_slots,  slots, attns_vis, mse = model(images)
    print(slots.shape, attns_vis.shape)
    print(mse)
    print('unit test passed')

if __name__ == '__main__':
    main()