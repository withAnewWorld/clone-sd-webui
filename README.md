# 환경설정
```bash
conda create -n {env_name} python=3.7
conda activate {env_name}
pip install -r requirements/prod.txt
cd latent-diffusion/
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg
```