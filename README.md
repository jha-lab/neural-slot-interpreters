#  [TMLR 2025] Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)
![CUDA](https://img.shields.io/badge/cuda-v11.1.1-76b900)
![License](https://img.shields.io/badge/license-Clear%20BSD-green)

This is the official repository for the paper [Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations](https://arxiv.org/abs/2403.07887).

![NSI Architecture](nsi.png)


## Table of Contents

- [Environment setup](#environment-setup)
- [Building the CLEVRTex Schema](#building-the-clevrtex-schema)
- [Model](#model)
  - [Image Encoder](#image-encoder)
  - [Schema Encoder](#schema-encoder)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Environment setup


The following shell script creates an anaconda environment called "nsi" and installs all the required packages. 
```shell
source env_setup.sh
```

## Building the CLEVRTex Schema

To build the CLEVRTex dataset, follow these steps:

1. Navigate to the `datasets` directory:
   ```shell
   cd datasets
   ```

2. Run the build script to download and extract the dataset:
   ```shell
   bash build_clevrtex.sh
   ```

This script will:
- Download the CLEVRTex dataset (~93GB)
- Extract the dataset files
- Process the dataset to create:
  - `program_space.json`: Contains the space of object properties (shapes, sizes, materials)
  - `program2label.json` and `label2program.json`: Mapping between program values and labels
  - `train_labels.npy`, `val_labels.npy`, `test_labels.npy`: Schema annotations
  - `images.h5`: Processed images for training, validation, and testing

The dataset will be split into:
- Training: 37,500 images
- Validation: 2,500 images
- Testing: 10,000 images

Each image contains objects with properties including size, shape, material, and 3D coordinates.

## Model
Our model consists of two main encoder components:

### Image Encoder

The ImageEncoder (located in `models/dino_encoder.py`) processes images using a Vision Transformer (ViT) backbone with DINO pre-training:

1. **Backbone**: Uses DINO ViT-Base with patch size 8 to extract visual features
2. **Slot Attention**: Processes the extracted features to discover object-centric representations
   - Takes the ViT features and decomposes them into a set of slots
   - Uses iterative attention mechanism over multiple rounds
3. **Decoder**: BroadcastDecoder reconstructs the original features from slots
   - Uses positional encoding and MLP to generate features and attention masks
   - Produces visualizations of slot attention masks
4. **Projection**: Projects slots to a common embedding space for alignment with program representations

The ImageEncoder outputs:
- Projected slots for alignment with program embeddings
- Attention visualizations showing object segmentation
- Raw slots containing object-centric information
- Reconstruction MSE for training supervision

### Schema Encoder

The Schema Encoder (located in `models/schema_encoder.py') processes structured schema representing object properties:

1. **Property Embeddings**: Separate dictionary embeddings for different object properties
   - Size embedding (3 possible sizes)
   - Shape embedding (4 possible shapes)
   - Material embedding (60 possible materials)
   - Position embedding (3D coordinates normalized to [0,1])
2. **Program Integration**: Combines all property embeddings into a unified representation
3. **Transformer Encoder**: Processes the program embeddings to capture relationships
4. **Projection**: Projects program representations to the same embedding space as image slots

Both encoders project their outputs to a common embedding space, enabling alignment between visual slots and program descriptions through contrastive learning.


## Training

The training process is implemented in `trainer/train_nsi.py` 

### Training Process

The training script (`trainer/train_nsi.py`) implements:

- **Dual Encoder Training**: Trains both image and schema encoders end-to-end using:
  - Contrastive loss between slot and program embeddings
  - Reconstruction loss for slot attention
- **Optimization**:
  - Adam optimizer with linear warmup
  - Learning rate decay when validation loss stagnates
  - Gradient clipping for stability
- **Monitoring**:
  - Tensorboard logging for losses and visualizations
  - Checkpointing of best models based on validation loss
  - Attention mask visualizations for slot assignments


### Key Parameters

The training script supports numerous parameters that can be adjusted:

- **Training Configuration**:
  - `--batch_size`: Number of samples per batch (default: 128)
  - `--epochs`: Total training epochs (default: 300)
  - `--lr`: Learning rate (default: 4e-4)
  - `--lr_warmup_steps`: Steps for linear warmup (default: 10000)
  - `--grad_clip`: Gradient clipping value (default: 1.0)
  - `--patience`: Epochs before learning rate decay (default: 4)

- **Model Architecture**:
  - `--num_slots`: Number of object slots (default: 15)
  - `--slot_dim`: Dimension of each slot (default: 192)
  - `--num_iterations`: Slot attention iterations (default: 5)
  - `--num_blocks`: Transformer encoder blocks (default: 8)
  - `--d_model`: Model dimension (default: 192)
  - `--num_heads`: Attention heads (default: 8)
  - `--dropout`: Dropout rate (default: 0.1)

- **Loss Function**:
  - `--tau`: Temperature for contrastive loss (default: 0.1)
  - `--beta`: Weight for reconstruction loss (default: 1.0)

- **Data Processing**:
  - `--image_size`: Input image resolution (default: 224)
  - `--max_program_len`: Maximum schema length (default: 10)

### Example Commands

Basic training with default parameters:

```shell
cd trainer/
python -u train_nsi.py --epochs 900 --batch_size 128 --num_blocks 8 --num_heads 8 --num_iterations 3 --data_path ../datasets/
```

## Evaluation Notebook

We provide an evaluation notebook (`notebooks/interpret_slots_clevrtex.ipynb`) that demonstrates how to visualize the learned slots and their alignment with object properties. This notebook shows how to:

- Load trained models and retrieve top-k images for specific slots
- Visualize how different slots correspond to specific object attributes
- Analyze the relationship between slots and object properties in the CLEVRTEX dataset
- Interpret what information each slot is encoding about the scene


## How to add a new task

Setting up a new task is easy, and mainly involves three steps.

- First, organize the annotations from the scene labels into a structured object-wise schema as demonstrated in `datasets/build_clevrtex_data.py`. This involves converting raw scene descriptions into a standardized format where each object's properties (e.g. size, shape, material, position) are encoded numerically.
- Second, design a schema encoder similar to the one in `models/program_encoder.py`. This encoder should:
  - Define appropriate embeddings for each property in your schema (like the size, shape, and material embeddings in CLEVRTexProgramEncoder)
  - Combine these embeddings into a unified representation
  - Use a transformer architecture to capture relationships between objects
  - Project the encoded schema into the embedding space used by the slot attention mechanism
- Finally, build an appropriate dataloader to jointly fetch the image and the corresponding schema as demonstrated in `dataset_loader/clevrtex_loader.py`. This dataloader should:
  - Load both image data and schema annotations
  - Apply any necessary transformations to the images
  - Handle padding for variable-length schemas
  - Return properly formatted tensors for both the images and schemas
  - Ensure proper batching and length information is maintained

The dataloader will bridge your schema representation with the visual data, allowing the model to learn the correspondence between visual slots and structured object descriptions.

## Citations

Please cite the paper and star this repo if you find it useful, thanks! Feel free to contact bdedhia@princeton.edu or open an issue if you have any questions. 
Cite our work using the following bitex entry:
```bibtex
@misc{dedhia2025neuralslotinterpretersgrounding,
      title={Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations}, 
      author={Bhishma Dedhia and Niraj K. Jha},
      year={2025},
      eprint={2403.07887},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.07887}, 
}
```

## License

The Clear BSD License
Copyright (c) 2025, Bhishma Dedhia and Jha Lab.
All rights reserved.

See License file for more details.