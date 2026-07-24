# -*- coding: utf-8 -*-
"""
第 2 周：图像处理进阶 — 滤波、边缘检测与形态学
=================================================

学习目标
--------
- 掌握灰度变换（线性/分段/对数/伽马）
- 理解直方图均衡化（全局 & CLAHE）
- 熟悉平滑滤波（均值、高斯、中值、双边）
- 实现边缘检测（Sobel、Laplacian、Canny）
- 掌握形态学操作（腐蚀/膨胀/开运算/闭运算/梯度）
- 学会轮廓提取与绘制

代码实践
--------
1. 读取输入图片，转为灰度图
2. 灰度变换：反转、对数变换、伽马校正
3. 直方图均衡化：全局均衡化 + CLAHE
4. 平滑滤波：均值、高斯、中值、双边滤波对比
5. 边缘检测：Sobel、Laplacian、Canny 对比
6. 形态学操作：腐蚀、膨胀、开运算、闭运算、形态学梯度
7. 轮廓检测与绘制
8. 所有结果输出到 ``results/`` 目录

运行方式
--------
.. code-block:: bash

    cd cv-dl-llm-summer-study
    python image_processing/week02_cv_demo.py

依赖
----
- Python 3.10+
- opencv-python
- numpy
- matplotlib
"""

import os
import sys
from pathlib import Path

# 强制 stdout 使用 UTF-8，避免 GBK 编码错误
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免无 GUI 时崩溃
import matplotlib.pyplot as plt

# 配置中文字体（Windows 上优先使用 Microsoft YaHei / SimHei）
if sys.platform == "win32":
    for _font in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        try:
            matplotlib.font_manager.findfont(_font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

# ---------------------------------------------------------------------------
# 0. 路径设置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT.parent / "ImageProcessing-Python-master"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 选取一张图片用于演示（优先用 Lena，经典测试图）
DEFAULT_IMAGE = INPUT_DIR / "blog01-start" / "Lena.png"

# ---------------------------------------------------------------------------
# 1. 工具函数
# ---------------------------------------------------------------------------


def imread_zh(path: Path) -> np.ndarray | None:
    """读取图片，支持含中文的路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_zh(path: Path, img: np.ndarray) -> bool:
    """保存图片，支持含中文的路径。"""
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img)
    elif ext == ".bmp":
        ok, buf = cv2.imencode(".bmp", img)
    else:
        ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))
        return True
    return False


def save_image(path: Path, img: np.ndarray):
    """保存图像并打印确认信息。"""
    ok = imwrite_zh(path, img)
    if ok:
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] 已保存: {path.name}  ({size_kb:.1f} KB)")
    else:
        print(f"  [ERR] 保存失败: {path}")


def make_comparison_grid(
    images: list[tuple[str, np.ndarray]],
    cols: int = 3,
    figsize_per_cell: tuple[float, float] = (4, 3.5),
    gray_cmap: bool = True,
) -> np.ndarray:
    """
    用 Matplotlib 生成对比网格图，返回 BGR 格式的 NumPy 数组。

    参数
    ----
    images : list of (title, img)
        每项为 (标题, 图像数组)。图像可以是灰度 (2D) 或彩色 (3D)。
    cols : int
        每行列数。
    figsize_per_cell : (w, h)
        每个子图的尺寸（英寸）。
    gray_cmap : bool
        灰度图是否使用 gray colormap。
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_cell[0], rows * figsize_per_cell[1]),
    )
    # 保证 axes 可迭代
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, (title, img) in enumerate(images):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray" if gray_cmap else None)
        else:
            # BGR -> RGB for matplotlib display
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # 隐藏多余的子图
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.canvas.draw()

    # 将 matplotlib figure 转为 OpenCV BGR 图像
    canvas = fig.canvas
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img_rgba = buf.reshape(h, w, 4)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr


# ---------------------------------------------------------------------------
# 2. 灰度变换
# ---------------------------------------------------------------------------

def demo_grayscale_transforms(gray: np.ndarray, stem: str):
    """
    演示灰度变换：反转、对数变换、伽马校正。

    - **反转**: s = 255 - r
    - **对数变换**: s = c * log(1 + r)，增强暗部细节
    - **伽马校正**: s = c * r^γ
        - γ < 1 → 变亮（增强暗部）
        - γ > 1 → 变暗（增强亮部）
    """
    print("\n" + "=" * 60)
    print("2. 灰度变换")
    print("=" * 60)

    # 2.1 反转（负片变换）
    inverted = 255 - gray
    print("  [反转] s = 255 - r  → 亮暗互换，类似胶片负片")

    # 2.2 对数变换（增强暗部细节）
    # c = 255 / log(1 + max)，使输出范围映射到 [0, 255]
    c_log = 255.0 / np.log(1 + gray.max())
    log_transformed = (c_log * np.log(1 + gray.astype(np.float64))).astype(np.uint8)
    print("  [对数] s = c * log(1+r)  → 压缩高亮区、扩展暗部")

    # 2.3 伽马校正
    # 归一化到 [0, 1]，做幂运算，再还原
    gray_norm = gray.astype(np.float64) / 255.0

    gamma_bright = 0.5  # γ < 1 → 整体变亮
    gamma_dark = 2.0    # γ > 1 → 整体变暗

    gamma_bright_img = (np.power(gray_norm, gamma_bright) * 255).astype(np.uint8)
    gamma_dark_img = (np.power(gray_norm, gamma_dark) * 255).astype(np.uint8)

    print(f"  [伽马 γ={gamma_bright}]  → 变亮，暗部细节提升")
    print(f"  [伽马 γ={gamma_dark}]    → 变暗，亮部细节提升")

    # 生成对比图
    comparison = make_comparison_grid([
        ("原始灰度", gray),
        ("反转 (Negative)", inverted),
        ("对数变换 (Log)", log_transformed),
        (f"伽马校正 (γ={gamma_bright})", gamma_bright_img),
        (f"伽马校正 (γ={gamma_dark})", gamma_dark_img),
    ], cols=3)

    save_image(OUTPUT_DIR / f"{stem}_01_grayscale_transforms.png", comparison)


# ---------------------------------------------------------------------------
# 3. 直方图均衡化
# ---------------------------------------------------------------------------

def demo_histogram(gray: np.ndarray, stem: str):
    """
    演示直方图均衡化：全局均衡化 & CLAHE（自适应直方图均衡化）。

    - **全局均衡化**: 拉伸像素分布，使直方图接近均匀
    - **CLAHE**: 分块做均衡化再拼接，避免过度放大噪声
    """
    print("\n" + "=" * 60)
    print("3. 直方图均衡化")
    print("=" * 60)

    # 3.1 全局直方图均衡化
    equ = cv2.equalizeHist(gray)
    print("  [全局均衡化] cv2.equalizeHist() → 直方图拉伸至近似均匀分布")

    # 3.2 CLAHE — 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    print("  [CLAHE] clipLimit=2.0, tile=8x8 → 局部均衡化，抑制噪声放大")

    # 绘制直方图对比
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    titles = ["原始灰度", "全局均衡化 (HE)", "CLAHE"]
    images = [gray, equ, clahe_img]

    for i, (title, img) in enumerate(zip(titles, images)):
        # 第一行：图像
        axes[0][i].imshow(img, cmap="gray")
        axes[0][i].set_title(title, fontsize=10)
        axes[0][i].axis("off")

        # 第二行：直方图
        axes[1][i].hist(img.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.8)
        axes[1][i].set_xlim([0, 256])
        axes[1][i].set_xlabel("像素值")
        if i == 0:
            axes[1][i].set_ylabel("频数")

    fig.tight_layout()
    fig.canvas.draw()
    canvas = fig.canvas
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img_bgr = cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    save_image(OUTPUT_DIR / f"{stem}_02_histogram_equalization.png", img_bgr)

    return equ, clahe_img


# ---------------------------------------------------------------------------
# 4. 平滑滤波（去噪）
# ---------------------------------------------------------------------------

def demo_smoothing(gray: np.ndarray, stem: str):
    """
    演示四种平滑滤波器。

    | 滤波器 | 原理 | 特点 |
    |--------|------|------|
    | 均值滤波 | 邻域像素取平均 | 简单快速，但会模糊边缘 |
    | 高斯滤波 | 邻域像素加权平均（中心权重大） | 比均值更自然，σ 控制平滑程度 |
    | 中值滤波 | 邻域像素取中位数 | 对椒盐噪声效果好，保留边缘 |
    | 双边滤波 | 空间+像素值双重加权 | 保边去噪，速度较慢 |
    """
    print("\n" + "=" * 60)
    print("4. 平滑滤波（去噪）")
    print("=" * 60)

    # 4.1 均值滤波 — 3x3 / 7x7
    blur_3x3 = cv2.blur(gray, (3, 3))
    blur_7x7 = cv2.blur(gray, (7, 7))
    print("  [均值 3x3] cv2.blur() → 轻微模糊")
    print("  [均值 7x7] cv2.blur() → 明显模糊，边缘也模糊了")

    # 4.2 高斯滤波 — 5x5, σ_x=0（自动从核大小推算）
    gauss_3x3 = cv2.GaussianBlur(gray, (3, 3), 0)
    gauss_7x7 = cv2.GaussianBlur(gray, (7, 7), 0)
    print("  [高斯 3x3] cv2.GaussianBlur() → 中心加权，比均值自然")
    print("  [高斯 7x7] cv2.GaussianBlur() → 更强平滑，仍比均值自然")

    # 4.3 中值滤波
    median_3x3 = cv2.medianBlur(gray, 3)
    median_7x7 = cv2.medianBlur(gray, 7)
    print("  [中值 3x3] cv2.medianBlur() → 对椒盐噪声极有效")
    print("  [中值 7x7] cv2.medianBlur() → 强去噪，保留一定边缘")

    # 4.4 双边滤波 — 保边去噪
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    print("  [双边] cv2.bilateralFilter(d=9,σC=75,σS=75) → 保边去噪")

    comparison = make_comparison_grid([
        ("原始灰度", gray),
        ("均值 3x3", blur_3x3),
        ("均值 7x7", blur_7x7),
        ("高斯 3x3", gauss_3x3),
        ("高斯 7x7", gauss_7x7),
        ("中值 3x3", median_3x3),
        ("中值 7x7", median_7x7),
        ("双边 (Bilateral)", bilateral),
    ], cols=4)

    save_image(OUTPUT_DIR / f"{stem}_03_smoothing.png", comparison)

    # 返回有代表性的结果供后续使用
    return gauss_3x3


# ---------------------------------------------------------------------------
# 5. 边缘检测
# ---------------------------------------------------------------------------

def demo_edge_detection(gray: np.ndarray, smoothed: np.ndarray, stem: str):
    """
    演示边缘检测算子：Sobel、Laplacian、Canny。

    - **Sobel**: 一阶导数，分别计算 x 和 y 方向的梯度
    - **Laplacian**: 二阶导数，对噪声敏感
    - **Canny**: 多阶段（梯度计算 → 非极大值抑制 → 双阈值 → 边缘连接），最常用
    """
    print("\n" + "=" * 60)
    print("5. 边缘检测")
    print("=" * 60)

    # 5.1 Sobel 算子 — 计算 x / y 方向梯度
    # cv2.Sobel 的 ddepth 用 CV_64F 避免负数截断
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # 取绝对值并转为 uint8
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # 合并 x 和 y 方向的梯度
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    print("  [Sobel X]  检测竖直边缘（水平梯度）")
    print("  [Sobel Y]  检测水平边缘（竖直梯度）")
    print("  [Sobel XY] X + Y 合并 → 全方向边缘")

    # 5.2 Laplacian 算子 — 二阶导数
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    print("  [Laplacian] 二阶导数 → 对噪声敏感，建议先平滑")

    # 5.3 Canny 边缘检测 — 最常用的边缘检测算法
    # 双阈值：低于 low → 非边缘；高于 high → 强边缘；中间 → 仅当与强边缘相连时保留
    canny_low = cv2.Canny(gray, 50, 150)
    canny_high = cv2.Canny(gray, 100, 200)
    # 对平滑后的图像做 Canny（更少噪声边缘）
    canny_smoothed = cv2.Canny(smoothed, 50, 150)

    print("  [Canny 50/150]  低阈值 → 更多边缘")
    print("  [Canny 100/200] 高阈值 → 仅强边缘")
    print("  [Canny 平滑后]  先高斯再去 Canny → 推荐的稳定做法")

    comparison = make_comparison_grid([
        ("原始灰度", gray),
        ("Sobel X", sobel_x_abs),
        ("Sobel Y", sobel_y_abs),
        ("Sobel XY 合并", sobel_combined),
        ("Laplacian", laplacian_abs),
        ("Canny (50/150)", canny_low),
        ("Canny (100/200)", canny_high),
        ("高斯平滑 → Canny (50/150)", canny_smoothed),
    ], cols=4)

    save_image(OUTPUT_DIR / f"{stem}_04_edge_detection.png", comparison)

    return canny_smoothed


# ---------------------------------------------------------------------------
# 6. 形态学操作
# ---------------------------------------------------------------------------

def demo_morphology(gray: np.ndarray, stem: str):
    """
    演示形态学操作。

    核心概念：
    - **结构元素 (kernel)**：定义"邻域"形状（矩形/椭圆/十字）
    - **腐蚀 (Erode)**：局部最小值 → 白色区域缩小，去除小噪点
    - **膨胀 (Dilate)**：局部最大值 → 白色区域扩大，填充小孔洞
    - **开运算 (Open)**：先腐蚀后膨胀 → 去除小物体/噪点
    - **闭运算 (Close)**：先膨胀后腐蚀 → 填充小孔洞/裂缝
    - **形态学梯度 (Gradient)**：膨胀 - 腐蚀 → 提取物体轮廓
    - **顶帽 (Top Hat)**：原图 - 开运算 → 提取亮的小结构
    - **黑帽 (Black Hat)**：闭运算 - 原图 → 提取暗的小结构
    """
    print("\n" + "=" * 60)
    print("6. 形态学操作")
    print("=" * 60)

    # 先二值化，便于观察形态学效果
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print("  [二值化] 阈值=127 → 为形态学操作准备二值图")

    # 定义结构元素
    # 椭圆核比矩形核更平滑
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # 6.1 腐蚀 & 膨胀
    eroded = cv2.erode(binary, kernel_ellipse, iterations=1)
    dilated = cv2.dilate(binary, kernel_ellipse, iterations=1)
    print("  [腐蚀] cv2.erode()  → 白色区域缩小，细小白点消失")
    print("  [膨胀] cv2.dilate() → 白色区域扩大，小黑孔被填充")

    # 6.2 开运算 & 闭运算
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_ellipse)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_ellipse)
    print("  [开运算] 先腐蚀后膨胀 → 去除小噪点、断开窄连接")
    print("  [闭运算] 先膨胀后腐蚀 → 填充小黑洞、连接邻近物体")

    # 6.3 形态学梯度
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_ellipse)
    print("  [梯度] 膨胀-腐蚀 → 提取物体边界")

    # 6.4 顶帽 & 黑帽
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel_ellipse)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel_ellipse)
    print("  [顶帽] 原图-开运算 → 提取亮的小区域")
    print("  [黑帽] 闭运算-原图 → 提取暗的小区域")

    # 6.5 多次迭代的效果演示
    eroded_2x = cv2.erode(binary, kernel_ellipse, iterations=2)
    dilated_2x = cv2.dilate(binary, kernel_ellipse, iterations=2)
    print("  [腐蚀 x2] 迭代2次 → 更强烈的腐蚀效果")
    print("  [膨胀 x2] 迭代2次 → 更强烈的膨胀效果")

    comparison = make_comparison_grid([
        ("二值图", binary),
        ("腐蚀 (Erode)", eroded),
        ("膨胀 (Dilate)", dilated),
        ("开运算 (Open)", opened),
        ("闭运算 (Close)", closed),
        ("形态学梯度 (Gradient)", gradient),
        ("顶帽 (Top Hat)", tophat),
        ("黑帽 (Black Hat)", blackhat),
        ("腐蚀 ×2", eroded_2x),
        ("膨胀 ×2", dilated_2x),
    ], cols=5)

    save_image(OUTPUT_DIR / f"{stem}_05_morphology.png", comparison)

    # 展示不同的结构元素
    kernel_demo = make_comparison_grid([
        ("二值图", binary),
        ("矩形核 腐蚀", cv2.erode(binary, kernel_rect, iterations=1)),
        ("椭圆核 腐蚀", cv2.erode(binary, kernel_ellipse, iterations=1)),
        ("十字核 腐蚀", cv2.erode(binary, kernel_cross, iterations=1)),
        ("矩形核 膨胀", cv2.dilate(binary, kernel_rect, iterations=1)),
        ("椭圆核 膨胀", cv2.dilate(binary, kernel_ellipse, iterations=1)),
        ("十字核 膨胀", cv2.dilate(binary, kernel_cross, iterations=1)),
    ], cols=4)

    save_image(OUTPUT_DIR / f"{stem}_06_kernel_comparison.png", kernel_demo)


# ---------------------------------------------------------------------------
# 7. 轮廓检测
# ---------------------------------------------------------------------------

def demo_contours(gray: np.ndarray, stem: str):
    """
    演示轮廓检测与绘制。

    流程：
    1. 二值化（或 Canny 边缘检测）得到边缘图
    2. cv2.findContours() 查找轮廓
    3. cv2.drawContours() 绘制轮廓
    """
    print("\n" + "=" * 60)
    print("7. 轮廓检测与绘制")
    print("=" * 60)

    # 7.1 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 7.2 查找轮廓
    # cv2.findContours 返回 contours, hierarchy
    # RETR_TREE: 提取所有轮廓并建立层级关系
    # CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线段，只保留端点
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"  [轮廓数] 共检测到 {len(contours)} 个轮廓")
    print("  [方法] RETR_TREE (全层级) + CHAIN_APPROX_SIMPLE (压缩)")

    # 7.3 在彩色原图上绘制轮廓
    # 恢复彩色原图用于绘制
    img_color_path = DEFAULT_IMAGE
    if img_color_path.exists():
        img_color = imread_zh(img_color_path)
    else:
        # fallback: 把灰度复制三通道
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 绘制所有轮廓（绿色，线宽 1）
    contour_all = img_color.copy()
    cv2.drawContours(contour_all, contours, -1, (0, 255, 0), 1)
    print("  [全部轮廓] 绿色细线绘制所有轮廓")

    # 绘制面积前 20 的轮廓（红色，线宽 2）
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    contour_top20 = img_color.copy()
    cv2.drawContours(contour_top20, contours_sorted[:20], -1, (0, 0, 255), 2)
    print("  [Top-20] 红色粗线绘制面积最大的 20 个轮廓")

    # 绘制轮廓 + 外接矩形
    contour_rect = img_color.copy()
    rect_count = 0
    for cnt in contours_sorted[:20]:
        area = cv2.contourArea(cnt)
        if area < 50:  # 过滤太小的轮廓
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_rect, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rect_count += 1
    print(f"  [外接矩形] 为面积 Top-{rect_count} 轮廓绘制蓝色矩形框")

    # 7.4 绘制凸包
    contour_hull = img_color.copy()
    for cnt in contours_sorted[:10]:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        hull = cv2.convexHull(cnt)
        cv2.drawContours(contour_hull, [hull], -1, (255, 255, 0), 2)
    print("  [凸包] 黄色绘制前10大轮廓的凸包")

    # 7.5 填充轮廓
    contour_filled = img_color.copy()
    for cnt in contours_sorted[:10]:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        cv2.drawContours(contour_filled, [cnt], -1, (0, 255, 255), cv2.FILLED)
    print("  [填充] 青色填充前10大轮廓内部")

    comparison = make_comparison_grid([
        ("二值图", binary),
        ("全部轮廓 (绿色)", contour_all),
        ("Top-20 轮廓 (红色)", contour_top20),
        ("外接矩形 (蓝色)", contour_rect),
        ("凸包 (黄色)", contour_hull),
        ("轮廓填充 (青色)", contour_filled),
    ], cols=3)

    save_image(OUTPUT_DIR / f"{stem}_07_contours.png", comparison)


# ---------------------------------------------------------------------------
# 8. 综合演示：去噪 → 边缘检测 完整流水线
# ---------------------------------------------------------------------------

def demo_pipeline(gray: np.ndarray, stem: str):
    """
    综合演示：完整的图像处理流水线。

    原始图 → 去噪 → 边缘检测 → 形态学细化 → 轮廓提取
    """
    print("\n" + "=" * 60)
    print("8. 综合流水线演示")
    print("=" * 60)

    # Step 1: 高斯去噪
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    print("  [1] 高斯去噪 (5x5)")

    # Step 2: Canny 边缘检测
    edges = cv2.Canny(denoised, 50, 150)
    print("  [2] Canny 边缘检测 (50/150)")

    # Step 3: 形态学闭运算连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    print("  [3] 形态学闭运算 (连接断裂边缘)")

    # Step 4: 查找并绘制显著轮廓
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipeline_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    significant = [c for c in contours if cv2.contourArea(c) > 50]
    cv2.drawContours(pipeline_result, significant, -1, (0, 255, 0), 2)
    print(f"  [4] 提取显著轮廓 (面积>50): {len(significant)} 个")

    comparison = make_comparison_grid([
        ("原始灰度", gray),
        ("高斯去噪", denoised),
        ("Canny 边缘", edges),
        ("闭运算后", edges_closed),
        ("最终轮廓", pipeline_result),
    ], cols=3)

    save_image(OUTPUT_DIR / f"{stem}_08_pipeline.png", comparison)


# ---------------------------------------------------------------------------
# 9. 椒盐噪声 + 去噪对比（额外演示）
# ---------------------------------------------------------------------------

def demo_denoising_comparison(gray: np.ndarray, stem: str):
    """
    添加椒盐噪声，然后用不同滤波器去噪，直观对比效果。
    """
    print("\n" + "=" * 60)
    print("9. 椒盐噪声去噪对比")
    print("=" * 60)

    # 添加椒盐噪声
    noisy = gray.copy()
    noise_ratio = 0.02  # 2% 的像素变为噪声
    num_salt = int(noise_ratio * gray.size * 0.5)
    num_pepper = int(noise_ratio * gray.size * 0.5)

    # 加盐噪声（白点）
    coords = [
        np.random.randint(0, i, num_salt) for i in gray.shape
    ]
    noisy[coords[0], coords[1]] = 255

    # 加椒噪声（黑点）
    coords = [
        np.random.randint(0, i, num_pepper) for i in gray.shape
    ]
    noisy[coords[0], coords[1]] = 0

    print(f"  [噪声] 添加 {noise_ratio*100:.0f}% 椒盐噪声（{num_salt}白 + {num_pepper}黑）")

    # 去噪对比
    mean_denoised = cv2.blur(noisy, (5, 5))
    gauss_denoised = cv2.GaussianBlur(noisy, (5, 5), 0)
    median_denoised = cv2.medianBlur(noisy, 5)
    bilateral_denoised = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

    print("  [均值 5x5]  噪声被模糊了，但边缘也被模糊")
    print("  [高斯 5x5]  比均值稍好，仍有残留噪声")
    print("  [中值 5x5]  椒盐噪声几乎完全消除，边缘保留好 ★推荐")
    print("  [双边 9x9]  保边能力强，但对椒盐噪声不如中值彻底")

    comparison = make_comparison_grid([
        ("原始灰度", gray),
        ("添加椒盐噪声 (2%)", noisy),
        ("均值滤波 (5x5)", mean_denoised),
        ("高斯滤波 (5x5)", gauss_denoised),
        ("中值滤波 (5x5) ★", median_denoised),
        ("双边滤波 (9x9)", bilateral_denoised),
    ], cols=3)

    save_image(OUTPUT_DIR / f"{stem}_09_denoising.png", comparison)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("第 2 周 -- 图像处理进阶")
    print("滤波 · 边缘检测 · 形态学 · 轮廓")
    print("=" * 60)

    # 验证输入目录
    if not INPUT_DIR.exists():
        print(f"\n[ERR] 输入目录不存在: {INPUT_DIR}")
        print("请确认 ImageProcessing-Python-master 位置正确。")
        return

    print(f"\n项目根目录 : {PROJECT_ROOT}")
    print(f"输入图片目录: {INPUT_DIR}")
    print(f"输出目录    : {OUTPUT_DIR}")

    # 读取图片
    if not DEFAULT_IMAGE.exists():
        print(f"\n[ERR] 找不到默认图片: {DEFAULT_IMAGE}")
        return

    stem = DEFAULT_IMAGE.stem  # "Lena"

    print(f"\n使用图片: {DEFAULT_IMAGE.name}")

    img = imread_zh(DEFAULT_IMAGE)
    if img is None:
        print("[ERR] 无法解码图片")
        return

    # 转为灰度图（所有后续操作的基础）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    print(f"灰度图尺寸: {w}×{h}, 像素范围: [{gray.min()}, {gray.max()}]\n")

    # ---- 按顺序执行各演示模块 ----

    # 2. 灰度变换
    demo_grayscale_transforms(gray, stem)

    # 3. 直方图均衡化
    equ, clahe_img = demo_histogram(gray, stem)

    # 4. 平滑滤波
    smoothed = demo_smoothing(gray, stem)

    # 5. 边缘检测
    edges = demo_edge_detection(gray, smoothed, stem)

    # 6. 形态学操作
    demo_morphology(gray, stem)

    # 7. 轮廓检测
    demo_contours(gray, stem)

    # 8. 综合流水线
    demo_pipeline(gray, stem)

    # 9. 椒盐噪声去噪
    demo_denoising_comparison(gray, stem)

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("[DONE] 第 2 周图像处理进阶完成！")
    print(f"       输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    outputs = sorted(OUTPUT_DIR.glob(f"{stem}_*"))
    if outputs:
        print("\n生成的文件:")
        for f in outputs:
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
