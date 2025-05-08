import os
import torch.nn as nn
import torch
from models.transformer import TransformerEncoder

def norm_pos(pos):
    '''
    renorm from [-3,3] to [0,1]
    '''
    pos = (pos+3)/6
    return pos


class CLEVRTexProgramEncoder(nn.Module):

    def __init__(self, num_blocks, d_model, num_heads, embed_dim, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_blocks, d_model, num_heads)
        self.sizes_size = 3
        self.shape_size = 4
        self.material_size = 60
        self.pos_coords = 3 
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.material_embedding = nn.Embedding(self.material_size, self.d_model)
        self.shape_embedding = nn.Embedding(self.shape_size, self.d_model)
        self.size_embedding = nn.Embedding(self.sizes_size, self.d_model)
        self.pos_embedding = nn.Sequential(
                nn.Linear(self.pos_coords, self.d_model),
                nn.ReLU()
        )
        self.program_embed = nn.Sequential( 
                nn.Linear(4*self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model)
        )
        self.embed_proj =  nn.Linear(self.d_model, self.embed_dim)
        self.norm_embed = nn.LayerNorm(self.embed_dim)
        self.embed_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4*self.embed_dim),
            nn.ReLU(),
            nn.Linear(4*self.embed_dim, self.embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, program):

        #embed program
        z_program = torch.cat([
            self.size_embedding(program[...,0].long()),
            self.shape_embedding(program[...,1].long()),
            self.material_embedding(program[...,2].long()),
            self.pos_embedding(norm_pos(program[...,3:6])),
        ], dim=-1)
        z_program = self.program_embed(z_program)
        z_program = self.encoder(z_program)
        z_program_proj = self.embed_proj(z_program)
        z_program_proj = z_program_proj + self.embed_mlp(self.norm_embed(z_program_proj))

        return z_program_proj, z_program
