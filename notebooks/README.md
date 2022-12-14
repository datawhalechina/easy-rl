# 蘑菇书附书代码

## 安装说明

目前支持Python 3.7和Gym 0.25.2版本。

创建Conda环境（需先安装Anaconda）
```bash
conda create -n joyrl python=3.7
conda activate joyrl
pip install -r requirements.txt
```

安装Torch：

```bash
# CPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU镜像安装
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
