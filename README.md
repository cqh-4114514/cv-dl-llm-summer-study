# CV-DL-LLM Summer Study

计算机视觉 / 深度学习入门 + 大模型基础 — 12 周学习计划

## 项目结构

```
cv-dl-llm-summer-study/
├── README.md
├── environment.yml              # 环境配置
├── notes/                       # 12 周学习笔记
│   ├── week01.md                # 第 1 周：Python 与图像基础
│   ├── week02.md
│   └── ...
├── image_processing/            # 第 1-2 周：图像处理
│   ├── week01_image_io.py
│   └── week02_cv_demo.py
├── deep_learning/               # 第 3-7 周：深度学习
├── llm_intro/                   # 第 8-10 周：大模型入门
├── cv_project/                  # 第 9-12 周：综合视觉项目
├── optional_extensions/         # 可选扩展
├── results/                     # 图表、训练曲线、输出结果
└── report/                      # 总结报告 / PPT
```

## 第 1 周：Python 与图像基础

### 运行方式

```bash
# 1. 激活环境
conda activate cv

# 2. 进入项目目录
cd cv-dl-llm-summer-study

# 3. 运行图像 I/O 练习
python image_processing/week01_image_io.py
```

### 输出说明

运行后在 `results/` 目录下生成每张输入图片对应的:
- `*_gray.png` — 灰度图
- `*_hsv.png` — HSV 颜色空间图
- `*_resized.png` — 缩放图（50%）
- `*_preview.png` — 四合一预览图

详细笔记见 [notes/week01.md](notes/week01.md)。


## 环境配置

```bash
conda env create -f environment.yml
```

主要依赖：Python 3.10+, OpenCV, NumPy, Matplotlib, PyTorch
