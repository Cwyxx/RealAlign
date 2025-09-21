# K-LoRA

Official implementation of [K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](https://arxiv.org/pdf/2502.18461).

![teaser](pictures/teaser.svg)


## 🔥 Examples
Below are the results of **K-LoRA**. The rows correspond to the respective style references, the columns correspond to the respective object references, and each cell represents the output obtained using a specific randomly selected seed.

![teaser](pictures/2.svg)
![teaser](pictures/520.svg)

## 🎨 Scaling factor
In the supplementary materials of our paper, we propose another scale s* . If you wish to generate images with more style block information, we recommend choosing s*. If you prefer more texture details, s is the better option. You can select based on your preferences. (For Flux, we recommend using s*.) Below are reference images for different scales.

![scale](pictures/scale.svg)


## 🚩TODO

- [x] super quick instruction for training local LoRAs
- [x] K-LoRA for SDXL (inference)
- [x] K-LoRA for FLUX (inference)
- [ ] k-LoRA for video models (inference)


## 🔧 Dependencies and Installation

### Installation
```
git clone https://github.com/ouyangziheng/K-LoRA.git
cd K-LoRA
pip install -r requirements.txt
```


### 1. Train LoRAs for subject/style images
In this step, 2 LoRAs for subject/style images are trained based on SDXL. Using SDXL here is important because they found that the pre-trained SDXL exhibits strong learning when fine-tuned on only one reference style image.

Fortunately, diffusers already implemented LoRA based on SDXL [here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md) and you can simply follow the instruction. 

For example, your training script would be like this.
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# for subject
export OUTPUT_DIR="lora-sdxl-dog"
export INSTANCE_DIR="dog"
export PROMPT="a sbu dog"
export VALID_PROMPT="a sbu dog in a bucket"

# for style
# export OUTPUT_DIR="lora-sdxl-waterpainting"
# export INSTANCE_DIR="waterpainting"
# export PROMPT="a cat of in szn style"
# export VALID_PROMPT="a man in szn style"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=8 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=50 \
  --seed="0" \
  --mixed_precision="no" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
```

* You can find style images in [aim-uofa/StyleDrop-PyTorch](https://github.com/aim-uofa/StyleDrop-PyTorch/tree/main/data).
* You can find content images in [google/dreambooth/tree/main/dataset](https://github.com/google/dreambooth/tree/main/dataset).


### 2. Inference

#### 2.1 Stable Diffusion 
You can directly use the script below for inference or interact by using the gradio.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LORA_PATH_CONTENT="..."
export LORA_PATH_STYLE="..."
export OUTPUT_FOLDER="..."  
export PROMPT="..."

python inference_sd.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --lora_name_or_path_content="$LORA_PATH_CONTENT" \
  --lora_name_or_path_style="$LORA_PATH_STYLE" \
  --output_folder="$OUTPUT_FOLDER" \
  --prompt="$PROMPT"

# using gradio 
# python inference_gradio.py \
#   --pretrained_model_name_or_path="$MODEL_NAME" \
#   --lora_name_or_path_content="$LORA_PATH_CONTENT" \
#   --lora_name_or_path_style="$LORA_PATH_STYLE" \
#   --output_folder="$OUTPUT_FOLDER" \
#   --prompt="$PROMPT"

```

#### 2.2 FLUX

If you want to test the **FLUX** version of K-LoRA, you can directly run the `inference_flux.py` script to perform inference using the community LoRA. 

If you are using **FLUX** for testing, it is recommended to use a higher version of Flux. Please refer to [FLUX](https://github.com/black-forest-labs/flux) for the dependency details.

If you wish to use the local **FLUX LoRA**, it is recommended to train it using the Dreambooth LoRA. For training instructions, you can refer to [dreambooth_lora](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora). 

For local LoRA inference, you can directly add the following plug-and-play command when performing inference.

```python
from utils import insert_community_flux_lora_to_unet

unet = insert_community_flux_lora_to_unet(
    unet=pipe,
    lora_weights_content_path=content_lora,
    lora_weights_style_path=style_lora,
    alpha=alpha,
    beta=beta,
    diffuse_step=flux_diffuse_step,
    content_lora_weight_name=content_lora_weight_name,
    style_lora_weight_name=style_lora_weight_name,
)
```

## Citation

If you use this code, please cite the following paper:
```BibTeX
@inproceedings{ouyang2025k,
  title={K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs},
  author={Ouyang, Ziheng and Li, Zhen and Hou, Qibin},
  booktitle={CVPR},
  year={2025}
}
```

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact the authors at [zihengouyang666@gmail.com](mailto:zihengouyang666@gmail.com).

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
