# 환경설정
```bash
conda create -n {env_name} python=3.8
conda activate {env_name}
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
cd latent-diffusion/
mkdir models/ldm/txt2img-f8-large
cd models/ldm/txt2img-f8-large
apt-get install axel
axel https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
pip install omegaconf>=2.0.0 pytorch-lightning>=1.7.7 torch-fidelity einops ldm-fix k-diffusion gradio pre-commit black
cd /clone-se-webui/
```
