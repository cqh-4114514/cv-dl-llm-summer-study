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

## 第 2 周：图像处理进阶

### 运行方式

```bash
# 1. 激活环境
conda activate cv

# 2. 进入项目目录
cd cv-dl-llm-summer-study

# 3. 运行图像处理进阶练习
python image_processing/week02_cv_demo.py
```

### 内容概览

| 模块 | 内容 |
|------|------|
| 灰度变换 | 反转、对数变换、伽马校正（γ=0.5 / 2.0） |
| 直方图均衡化 | 全局均衡化 (HE) + CLAHE 自适应均衡化 |
| 平滑滤波 | 均值、高斯、中值、双边滤波对比 |
| 边缘检测 | Sobel、Laplacian、Canny 算子对比 |
| 形态学操作 | 腐蚀/膨胀/开闭运算/梯度/顶帽/黑帽 |
| 轮廓提取 | 轮廓检测、外接矩形、凸包、填充 |
| 综合流水线 | 去噪→Canny→形态学→轮廓 完整流程 |
| 去噪对比 | 椒盐噪声 + 不同滤波器去噪效果对比 |

### 输出说明

运行后在 `results/` 目录下生成：
- `*_01_grayscale_transforms.png` — 灰度变换对比
- `*_02_histogram_equalization.png` — 直方图均衡化 + 直方图
- `*_03_smoothing.png` — 各种平滑滤波对比
- `*_04_edge_detection.png` — 各种边缘检测对比
- `*_05_morphology.png` — 形态学操作效果
- `*_06_kernel_comparison.png` — 不同结构元素对比
- `*_07_contours.png` — 轮廓提取效果
- `*_08_pipeline.png` — 综合流水线
- `*_09_denoising.png` — 椒盐噪声去噪对比

详细笔记见 [notes/week02.md](notes/week02.md)。

## 环境配置

```bash
conda env create -f environment.yml
```

主要依赖：Python 3.10+, OpenCV, NumPy, Matplotlib, PyTorch
