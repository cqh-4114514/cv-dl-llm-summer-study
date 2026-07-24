# 第 1 周：Python 与图像基础

> **学习日期**：2026-07-24  
> **参考资源**：[Python 图像处理与识别](https://github.com/eastmountyxz/ImageProcessing-Python) · [华为云博客](https://bbs.huaweicloud.com/blogs/336932)

---

## 1. 图像是什么？

图像在计算机中是一个**多维数组（矩阵）**。每个元素叫一个**像素（pixel）**。

| 概念 | 说明 |
|------|------|
| **像素 (Pixel)** | 图像的最小单元，存储颜色/亮度信息 |
| **分辨率 (Resolution)** | 图像的宽度 × 高度（如 1920×1080） |
| **通道 (Channel)** | 每个像素的颜色分量数：灰度=1，RGB/BGR=3，RGBA=4 |
| **位深 (Bit Depth)** | 每个通道的比特数，常见为 8bit（0–255） |

在 NumPy 中，一张 512×512 的彩色图像就是一个 `shape=(512, 512, 3)` 的 `uint8` 数组。

```python
import cv2
img = cv2.imread("lena.png")
print(type(img))   # <class 'numpy.ndarray'>
print(img.shape)   # (512, 512, 3)
print(img.dtype)   # uint8
```

---

## 2. 颜色空间

### 2.1 RGB vs BGR

| 标准 | 通道顺序 | 常用库 |
|------|----------|--------|
| **RGB** | Red → Green → Blue | Matplotlib, Pillow |
| **BGR** | Blue → Green → Red | OpenCV（默认） |

> ⚠️ OpenCV 读取图片得到的是 BGR 格式，用 Matplotlib 显示前需要转换：
> `img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

### 2.2 灰度图 (Grayscale)

每个像素只有一个亮度值（0–255），不包含颜色信息。

$$Gray = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
```

### 2.3 HSV 颜色空间

- **H (Hue 色调)**：颜色类型，0–179（OpenCV 中），对应红橙黄绿青蓝紫
- **S (Saturation 饱和度)**：颜色的纯度，0–255。0=灰色，255=最鲜艳
- **V (Value 明度)**：亮度，0–255。0=全黑，255=最亮

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

HSV 的优势：将**颜色**和**亮度**解耦，便于做颜色阈值、物体追踪等。

---

## 3. 图像基本操作

### 3.1 读取

| 方法 | 说明 |
|------|------|
| `cv2.imread(path)` | 返回 BGR 格式的 NumPy 数组 |
| `cv2.imread(path, 0)` | 直接读取为灰度图 |

### 3.2 显示

```python
cv2.imshow("window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3.3 保存

```python
cv2.imwrite("output.png", img)
```

### 3.4 缩放

```python
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
```

常用插值方法：

| 方法 | 适用场景 |
|------|----------|
| `INTER_AREA` | 缩小图片（推荐） |
| `INTER_LINEAR` | 默认，通用双线性插值 |
| `INTER_CUBIC` | 放大图片（较慢但质量好） |

### 3.5 图像属性速查

```python
h, w = img.shape[:2]       # 高度、宽度
channels = img.shape[2]    # 通道数（灰度图无此维度）
total_pixels = img.size    # 总像素数 = H × W × C
dtype = img.dtype          # 数据类型，通常 uint8
vmin, vmax = img.min(), img.max()  # 像素值范围
```

---

## 4. NumPy 与图像的桥梁

图像就是 NumPy 数组，可以：

```python
# 切片裁图
roi = img[50:200, 100:300]

# 分离通道
b, g, r = cv2.split(img)

# 创建纯色图
blue_img = np.zeros((256, 256, 3), dtype=np.uint8)
blue_img[:, :, 0] = 255  # B 通道全开 → 纯蓝
```

---

## 5. 本周实验输出

运行 `week01_image_io.py` 后在 `results/` 目录下生成每张输入图片对应的：

| 文件后缀 | 说明 |
|----------|------|
| `*_gray.png` | 灰度图（单通道） |
| `*_hsv.png` | HSV 颜色空间图（用于显示时自动转回 BGR） |
| `*_resized.png` | 缩放图（缩小为原始的 50%） |
| `*_preview.png` | 四合一预览拼接图（原图 + 灰度 + HSV + 缩放） |

每条输出含义：

```
[Lena (原始 BGR)]
  尺寸 (H×W) : 512 × 512      ← 图片分辨率
  通道数      : 3               ← B、G、R 三个通道
  像素值范围  : [0, 255]        ← 8bit 图像的标准范围
  数据类型    : uint8           ← 无符号 8 位整数
```

- **uint8** 范围是 0–255，255=纯白/最亮
- 如果出现 0–1 的 float 范围，说明图片已归一化（`dtype=float32/64`）
- HSV 图中 H 通道范围是 0–179，S 和 V 是 0–255

---

## 6. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 图像偏蓝/颜色不对 | RGB/BGR 通道搞反 | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| 图片读进来是 `None` | 路径错误或中文路径 | 检查路径、不要用中文目录名 |
| 保存的灰度图仍是彩色 | 保存时未转灰度 | 灰度图应 `shape=(H,W)` 不含第 3 维 |
| `matplotlib` 显示图片颜色奇怪 | 输入了 BGR 而非 RGB | 先用 `cvtColor` 转换 |

---

## 7. 下周预告

第 2 周将学习：灰度变换、直方图均衡、平滑去噪、Sobel/Canny 边缘检测、形态学操作（腐蚀/膨胀/开闭运算）和轮廓提取。
