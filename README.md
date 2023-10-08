# 환경설정
```bash
conda create -n {env_name} python=3.7
conda activate {env_name}
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
cd latent-diffusion/
wget -O models/ldm/text2img256/model.zip https://ommer-lab.com/files/latent-diffusion/text2img.zip
cd models/ldm/text2img256
unzip -o model.zip
pip install omegaconf>=2.0.0 pytorch-lightning>=1.7.7 torch-fidelity einops ldm-fix k-diffusion gradio
```
