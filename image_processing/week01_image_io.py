# -*- coding: utf-8 -*-
"""
第 1 周：Python 与图像基础 — 图像 I/O 练习
============================================

学习目标
--------
- 使用 OpenCV / NumPy / Matplotlib 读写和显示图像
- 理解 NumPy 数组、像素、分辨率、灰度图、RGB/BGR/HSV 颜色空间
- 掌握图像的尺寸、通道数、像素值范围等基本属性

代码实践
--------
1. 读取至少 3 张不同格式/内容的图片
2. 转换并保存：灰度图、HSV 图、缩放图
3. 输出每张图的：尺寸 (height x width)、通道数、像素值范围 (min/max)
4. 所有输出图像保存在 ``results/`` 目录中

运行方式
--------
.. code-block:: bash

    cd cv-dl-llm-summer-study
    python image_processing/week01_image_io.py

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
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. 路径设置
# ---------------------------------------------------------------------------
# 项目根目录（cv-dl-llm-summer-study/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 输入图片目录 —— 复用 ImageProcessing-Python-master 中的示例图片
INPUT_DIR = PROJECT_ROOT.parent / "ImageProcessing-Python-master"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 选取 3 张不同来源的图片（不同格式：png / png / jpg）
IMAGE_PATHS = [
    INPUT_DIR / "blog01-start" / "Lena.png",      # PNG 经典测试图 (413x412)
    INPUT_DIR / "blog01-start" / "flower.png",     # PNG 花卉图
    INPUT_DIR / "blog01-start" / "testyxz.jpg",    # JPG 风景图
]

# ---------------------------------------------------------------------------
# 1. 工具函数
# ---------------------------------------------------------------------------
def imread_zh(path: Path) -> np.ndarray | None:
    """
    读取图片——支持含中文的路径。
    OpenCV 的 cv2.imread() 在 Windows 上无法处理非 ASCII 路径，
    因此使用 np.fromfile + cv2.imdecode 的替代方案。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_zh(path: Path, img: np.ndarray) -> bool:
    """
    保存图片——支持含中文的路径。
    OpenCV 的 cv2.imwrite() 在 Windows 上无法处理非 ASCII 路径，
    因此使用 cv2.imencode + tofile 的替代方案。
    """
    ext = path.suffix.lower()
    # 根据扩展名确定编码格式
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


def image_info(name: str, img: np.ndarray):
    """打印图像的尺寸、通道数、像素值范围"""
    if img is None:
        print(f"[{name}] [ERR] 图片为空，跳过")
        return

    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    vmin, vmax = img.min(), img.max()
    dtype = img.dtype

    print(f"[{name}]")
    print(f"  尺寸 (H x W) : {h} x {w}")
    print(f"  通道数        : {ch}")
    print(f"  像素值范围    : [{vmin}, {vmax}]")
    print(f"  数据类型      : {dtype}")
    print()


def save_image(path: Path, img: np.ndarray):
    """保存图像并打印确认信息"""
    ok = imwrite_zh(path, img)
    if ok:
        size_kb = path.stat().st_size / 1024
        print(f"  [OK] 已保存: {path.name}  ({size_kb:.1f} KB)")
    else:
        print(f"  [ERR] 保存失败: {path}")


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("第 1 周 -- 图像 I/O 练习")
    print("=" * 60)

    # 打印关键路径
    print(f"\n项目根目录 : {PROJECT_ROOT}")
    print(f"输入图片目录: {INPUT_DIR}")
    print(f"输出目录    : {OUTPUT_DIR}")

    # 验证输入目录存在
    if not INPUT_DIR.exists():
        print(f"\n[ERR] 输入目录不存在！请确认 ImageProcessing-Python-master 位置正确。")
        return

    print()

    processed = 0

    for img_path in IMAGE_PATHS:
        # ---- 2.1 读取图片 ------------------------------------------------
        if not img_path.exists():
            print(f"[WARN] 跳过不存在的文件: {img_path}")
            continue

        stem = img_path.stem  # 文件名不带扩展名，如 "Lena"

        # 读取（封装了 np.fromfile + imdecode，支持中文路径）
        img = imread_zh(img_path)
        if img is None:
            print(f"[WARN] 无法解码: {img_path.name}")
            continue

        print(f"[>>>] 处理图片: {img_path.name}")
        processed += 1

        # ---- 2.2 输出原始属性 -------------------------------------------
        image_info(f"{stem} (原始 BGR)", img)

        # ---- 2.3 灰度图 -------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_info(f"{stem} (灰度)", gray)
        save_image(OUTPUT_DIR / f"{stem}_gray.png", gray)

        # ---- 2.4 HSV 图 -------------------------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_info(f"{stem} (HSV)", hsv)
        save_image(OUTPUT_DIR / f"{stem}_hsv.png", hsv)

        # ---- 2.5 缩放图（缩小为原来的一半）-------------------------------
        h, w = img.shape[:2]
        resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        image_info(f"{stem} (缩放 50%)", resized)
        save_image(OUTPUT_DIR / f"{stem}_resized.png", resized)

        # ---- 2.6 拼接预览图 --------------------------------------------
        # 生成一张包含原图、灰度、HSV、缩放的对比图
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)   # HSV -> BGR 才能正常显示
        resized_back = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)

        top = np.hstack([img, gray_3ch])
        bot = np.hstack([hsv_bgr, resized_back])
        preview = np.vstack([top, bot])

        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(preview, "Original", (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "Gray", (w + 10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "HSV", (10, h + 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "Resized 50%", (w + 10, h + 25), font, 0.7, (0, 255, 0), 2)

        save_image(OUTPUT_DIR / f"{stem}_preview.png", preview)

        print("-" * 40)

    # -------------------------------------------------------------------
    # 3. 总结输出
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"[DONE] 第 1 周图像 I/O 练习完成！共处理 {processed} 张图片。")
    print(f"       输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 列出输出目录中的所有文件
    outputs = sorted(OUTPUT_DIR.glob("*"))
    if outputs:
        print("\n生成的文件:")
        for f in outputs:
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
