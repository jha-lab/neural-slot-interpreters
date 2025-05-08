import os.path
import sys
sys.path.append('../')
import argparse
import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets, transforms, utils
from datetime import datetime
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_loader.clevrtex_loader import CLEVrTexProgramDataSet     
from models.dino_encoder import ImageEncoder
from models.program_encoder import CLEVrTexProgramEncoder
from utils import slot_program_score, SymmetricCELoss

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)

#trainer params
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=10000)
parser.add_argument('--grad_clip', type=float, default=1.0)

#paths
parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='../logs/clevrtex/')
parser.add_argument('--data_path', default='../datasets/clevrtex/')
parser.add_argument('--max_program_len', type=int, default=10)

#model params
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--hidden_dim', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_iterations', type=int, default=5)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--num_slot_heads', type=int, default=1)
parser.add_argument('--slot_dim', type=int, default=192)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--beta', type=float, default=1.0)

#loss params
parser.add_argument('--tau', type=float, default=0.1)


args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

transform = transforms.Compose(
        [
            transforms.Resize(args.image_size,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
        ]
)

train_dataset = CLEVrTexProgramDataSet(args.data_path, 'train', transform, max_program_len=args.max_program_len)
val_dataset = CLEVrTexProgramDataSet(args.data_path, 'val', transform, max_program_len=args.max_program_len)

train_sampler = None
val_sampler = None

def __collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    lengths = torch.stack(lengths)
    return imgs, labels, lengths

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn = __collate_fn, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, collate_fn = __collate_fn, **loader_kwargs)
train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5
slot_encoder = ImageEncoder(args)
program_encoder = CLEVrTexProgramEncoder(args.num_blocks, args.d_model, args.num_heads, args.embed_dim, dropout=args.dropout)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    stagnation_counter = checkpoint['stagnation_counter']
    lr_decay_factor = checkpoint['lr_decay_factor']
    slot_encoder.load_state_dict(checkpoint['slot_encoder'])
    program_encoder.load_state_dict(checkpoint['program_encoder'])

else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    stagnation_counter = 0
    lr_decay_factor = 1.0

slot_encoder = slot_encoder.cuda()
program_encoder = program_encoder.cuda()
optimizer = Adam(list(slot_encoder.parameters()) + list(program_encoder.parameters()), lr=args.lr)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

def linear_warmup(step, start_value, final_value, start_step, final_step):
    
    assert start_value <= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    
    return value

def visualize(image, attns, N=8):

    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    return torch.cat((image, attns), dim=1).view(-1, 3, H, W)


for epoch in range(start_epoch, args.epochs):
    
    slot_encoder.train()
    program_encoder.train()

    for batch, (images, labels, lengths) in enumerate(train_loader):

        global_step = epoch * train_epoch_size + batch

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        optimizer.param_groups[0]['lr'] = args.lr*lr_warmup_factor*lr_decay_factor

        images = images.cuda()
        labels = labels.cuda()

        #create a mask
        mask = torch.arange(args.max_program_len).reshape(1,-1).repeat(args.batch_size,1) < lengths[:,None]
        mask = mask.cuda()

        optimizer.zero_grad()
        slot, attns,_, _, mse = slot_encoder(images)
        program, _ = program_encoder(labels)
        logits = slot_program_score(slot, program, mask)
        contrastive_loss = SymmetricCELoss(logits, tau=args.tau)
        loss = contrastive_loss + args.beta*mse
        loss.backward()
        clip_grad_norm_(list(slot_encoder.parameters()), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} '.format(
                      epoch+1, batch, train_epoch_size, loss.item()))

                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/contrastive_loss', contrastive_loss.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                writer.add_scalar('TRAIN/lr_main', optimizer.param_groups[0]['lr'], global_step) 


    with torch.no_grad():

        vis_recon = visualize(images, attns, N=20)
        grid = vutils.make_grid(vis_recon, nrow=args.num_slots + 1, pad_value=0.2)[:, 2:-2, 2:-2]
        writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), grid)

    with torch.no_grad():

        slot_encoder.eval()
        program_encoder.eval()

        val_loss = 0.
        val_mse = 0.
        val_contrastive_loss = 0.

        for batch, (images, labels, lengths) in enumerate(val_loader):

            images = images.cuda()
            labels = labels.cuda()

            #create a mask
            mask = torch.arange(args.max_program_len).reshape(1,-1).repeat(args.batch_size,1) < lengths[:,None]
            mask = mask.cuda()

            slot, attns,_,  _, mse = slot_encoder(images)
            program, _ = program_encoder(labels)
            logits = slot_program_score(slot, program, mask)
            contrastive_loss = SymmetricCELoss(logits, tau=args.tau)
            loss = contrastive_loss + args.beta*mse
            val_loss += loss.item()
            val_mse += mse.item()
            val_contrastive_loss += contrastive_loss.item()

        val_loss = val_loss / val_epoch_size
        val_mse = val_mse / val_epoch_size
        val_contrastive_loss = val_contrastive_loss / val_epoch_size

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/contrastive_loss', val_contrastive_loss, epoch+1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)
        vis_recon = visualize(images, attns, N=20)
        grid = vutils.make_grid(vis_recon, nrow=args.num_slots + 1, pad_value=0.2)[:, 2:-2, 2:-2]
        writer.add_image('VAL_recon/epoch={:03}'.format(epoch+1), grid)
        

        print('===> Epoch: {:3}  \t Loss {:F}'.format(
                epoch+1, val_loss))
        
        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            best_epoch = epoch + 1
            stagnation_counter = 0
            model_dict = {}
            model_dict['slot_encoder'] = slot_encoder.state_dict()
            model_dict['program_encoder'] = program_encoder.state_dict()
            torch.save(model_dict, os.path.join(log_dir, 'best_model.pt'))

        else:
            stagnation_counter += 1
            if stagnation_counter >= args.patience:
                stagnation_counter = 0
                lr_decay_factor *= 0.5
        
        writer.add_scalar('VAL/best loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'stagnation_counter': stagnation_counter,
            'lr_decay_factor': lr_decay_factor,
            'optimizer': optimizer.state_dict(),
            'slot_encoder': slot_encoder.state_dict(),
            'program_encoder': program_encoder.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))
 
        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()