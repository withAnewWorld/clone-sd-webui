# 환경설정
```bash
conda create -n {env_name} python=3.7
conda activate {env_name}
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
cd latent-diffusion/
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
pip install omegaconf>=2.0.0 pytorch-lightning>=1.7.7 torch-fidelity einops ldm-fix k-diffusion gradio
```
