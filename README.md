# Style-Splat: Official Code Repository

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://bernard0047.github.io/stylesplat/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.09473-b31b1b.svg)](https://arxiv.org/abs/2407.09473)
## 🚀 Quick Start

### Prerequisites

- CUDA-compatible GPU
- Conda package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bernard0047/style-splat --recursive

2. Set up the Conda environment:

   ```bash
    conda create -n style-splat python=3.8 -y
    conda activate style-splat

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips

    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn

3. Install DEVA tracking:

   ```bash
    cd style-splat
    git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git
    cd Tracking-Anything-with-DEVA
    pip install -e .
    bash scripts/download_models.sh
    cd ../..

## 🎨 Usage

### 1. Preparing Pseudo Labels

Before training or applying style transfer, you need to prepare pseudo labels for your dataset.

1. Edit `scripts/prepare_pseudo_label.sh`:
   - Set the `DATASET_ROOT` variable to your dataset's root folder
   - Adjust the `SCALE` variable as needed for your dataset

   Example:
   ```bash
   DATASET_ROOT="/path/to/your/dataset"
   SCALE=2

2. Generate object masks:
   ```bash
    bash scripts/prepare_pseudo_label.sh
    ```
    - This will create an object_mask folder in your dataset directory.

### 2. Training Gaussian Grouping Instance
   - Train the model to group Gaussians:
        ```bash 
        python train.py -s data/your_scene -r 2 \
        --config config/gaussian_dataset/train.json --port 8000 \
        -m outputs/your_scene_output
        ```

### 3. Style Transfer

1. Create a configuration file:
    - Copy config/object_style_transfer/bear.json for your scene
    - Use Image Color Picker to select object IDs
    - Update select_obj_id in your configuration file
2. Run style transfer:
    - Use edit_object_style_transfer.py for single object
    - Use edit_object_style_transfer_multiple.py for multiple objects
    - See transfer_style.sh for example usage

## 📚 Additional Resources

- [DEVA Tracking](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- [Image Color Picker](https://imagecolorpicker.com): An online tool to help identify object IDs from your mask images.

## ✨ Acknowledgements

We would like to express our gratitude to the following projects that have significantly contributed to the development of Style-Splat:

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Gaussian-Grouping](https://github.com/lkeab/gaussian-grouping)

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---