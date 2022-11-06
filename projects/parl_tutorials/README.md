## 运行环境

由于```parl```和```paddle```容易与notebook相关模块发生版本冲突，因此推荐新建一个Conda环境：
```bash
conda create -n parl python=3.7
```

然后安装```parl```和```paddle```：
```bash
pip install parl==2.0.5

pip install paddlepaddle-gpu==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装其他依赖：
```bash
pip install -r parl_requirements.txt
```