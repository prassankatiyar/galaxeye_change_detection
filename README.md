# Binary Change Detection on EO-SAR Satellite Image Pairs




## Project Description

The task is to detect which buildings got damaged or destroyed after a disaster, using two satellite images taken of the same area at different times — a pre-event optical/EO photograph and a post-event SAR radar image. The output is a binary mask where 1 means the pixel corresponds to a damaged or destroyed building, and 0 means no change.

The main challenge is that the two images come from completely different types of sensors. The EO image is basically a colour photograph. The SAR image is a radar scan that works through clouds and at night, but looks nothing like a photograph. Combining them usefully is the core problem.

**My approach:** U-Net segmentation model with a ResNet-18 encoder pretrained on ImageNet. I concatenate the EO image (3 channels) and SAR image (1 channel) into a single 4-channel input — this is called early fusion. To handle the severe class imbalance (only about 1% of pixels are actually damaged), I used a combination of Dice loss + BCE loss and biased patch sampling during training.

---

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA 11.8 (tested on GTX 1650 Ti, 4 GB VRAM)
- All dependencies are listed in `requirements.txt` with pinned versions

Key libraries used:

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.0+cu118 | Deep learning framework |
| torchvision | 0.21.0+cu118 | Image utilities |
| segmentation-models-pytorch | 0.3.3 | U-Net implementation |
| albumentations | 1.3.1 | Data augmentation |
| tifffile | 2023.7.10 | Reading satellite TIFF files |
| numpy | 1.26.4 | Array operations |
| tqdm | 4.66.1 | Progress bars |
| pyyaml | 6.0.1 | Config file loading |
| matplotlib | 3.8.2 | Visualisation |
| tensorboard | 2.15.1 | Training monitoring |

---

## Environment Setup



```bash
# Step 1: Create a virtual environment
python -m venv galaxeye_env

# Step 2: Activate it
galaxeye_env\Scripts\activate

# Step 3: Install PyTorch with CUDA 11.8
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install everything else
pip install -r requirements.txt
```

```


---

## Dataset Structure

Place the dataset so the directory looks like this:


data/
├── train/
│   ├── pre-event/        Pre-disaster 
│   ├── post-event/       Post-disaster
│   └── target/           masks 
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Label values in target masks:

| Value | Meaning | Binary remapping |
|-------|---------|-----------------|
| 0 | Background (road, field, water) | 0 (no change) |
| 1 | Intact building | 0 (no change) |
| 2 | Damaged building | 1 (change) |
| 3 | Destroyed building | 1 (change) |

The code automatically remaps 4-class labels to binary during data loading.

---

## Training

Before running training, open `config.yaml` and update the data path:

```yaml
data:
  root: "D:/galaxeye_data"   # change this to where your dataset folder is
```

Then run:

```bash
python train.py --config config.yaml
```

Checkpoints are saved to `runs/unet_resnet18_earlyfusion_v1/`. The best checkpoint (highest validation IoU) is saved as `best.pth`.

On a GTX 1650 Ti with batch size 4, each epoch takes around 8-9 minutes. Full 40 epoch training is about 6 hours.



## Evaluation

Evaluate on the **test split:**

```bash
python eval.py --data_path /path/to/test --weights /path/to/checkpoint.pth --split test --out results_test.json
```

Evaluate on the **validation split:**

```bash
python eval.py --data_path /path/to/data --split val --weights /path/to/best.pth --out results_val.json
```


The script prints IoU, Precision, Recall, F1, and the confusion matrix.



To generate prediction visualisations:

```bash
python visualize.py --data_path /path/to/data --split val \
    --weights /path/to/best.pth --num 12
```

---

## Model Weights

The trained checkpoint (`best.pth`)  is hosted on Google Drive:

**[Download best.pth — Google Drive](https://drive.google.com/file/d/1NBIAC8B5sRVxfB2ea2aepsbhLb1a_2km/view?usp=sharing)**

After downloading, pass its path to `--weights` when running eval.py.

---

## Results

| Split | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Validation (334 images) | 0.2185 | 0.3131 | 0.4195 | 0.3586 |
| Test (77 images) | 0.0231 | 0.0381 | 0.0558 | 0.0452 |

The large gap between validation and test results comes mainly from scene_01 in the test split, which has cloud-covered EO images, partially masked tiles, and very sparse ground truth labels. On clear images from other scenes the model achieves IoU between 0.64 and 0.72. The technical report has a full failure analysis.

Confusion matrix on validation (pixel counts):

```
                    Predicted 0    Predicted 1
Actual 0            335,424,679      7,093,532
Actual 1              4,473,143      3,233,030
```

---

