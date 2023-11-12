# 환경설정
```bash
conda create -n {env_name} python=3.10
conda activate {env_name}
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt # sd v-1.5
pip install omegaconf>=2.0.0 pytorch-lightning>=1.7.7 torch-fidelity einops ldm-fix k-diffusion gradio pre-commit black
pip install transfomers
pre-commit install
apt-get install msttcorefonts # for font(arial.ttf)
```
