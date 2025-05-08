import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


def slot_program_score(slots, programs, mask=None):
    '''
    finds scores between all slots and programs
    slot = B x N x d
    program = B x T x d 
    mask = B x T
    '''
    B, N, d = slots.shape
    B, T, d = programs.shape
    slots = slots.reshape(B*N, d)
    programs = programs.reshape(B*T, d)
    scores = torch.mm(slots, programs.t()) # B*N x B*T
    scores = scores.reshape(B, N, B, T)
    scores = scores.permute(0, 2, 1, 3) # B x B x N x T
    scores = torch.max(scores,dim=2)[0] # B x B x T
    if mask is not None:
        mask = mask.unsqueeze(0) # 1 x B x T
        scores = scores*mask # B x B x T
    logits = scores.sum(dim=2) # B x B
    return logits


def SymmetricCELoss(sim_logits, tau=0.5):
    '''
    Given similarity logits, compute the symmetric cross entropy loss
    '''
    B, _ = sim_logits.shape
    sim_logits = sim_logits / tau
    labels = torch.arange(B).long().cuda()
    loss_ims = F.cross_entropy(sim_logits, labels)
    loss_progs = F.cross_entropy(sim_logits.transpose(0,1), labels)
    loss = (loss_ims + loss_progs)/2
    return loss


