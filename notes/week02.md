# 第 2 周：图像处理进阶 — 滤波、边缘检测与形态学

> **学习日期**：2026-07-24
> **参考资源**：[OpenCV 官方教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) · [ImageProcessing-Python](https://github.com/eastmountyxz/ImageProcessing-Python)

---

## 1. 灰度变换 (Gray-level Transformations)

灰度变换是对每个像素值应用一个函数 $s = T(r)$，改变图像的亮度和对比度。

### 1.1 图像反转 (Negative)

$$s = 255 - r$$

效果类似胶片负片——亮变暗、暗变亮。适用于增强嵌入在暗区的白色/灰色细节。

```python
inverted = 255 - gray
```

### 1.2 对数变换 (Log Transform)

$$s = c \cdot \log(1 + r), \quad c = \frac{255}{\log(1 + \max(r))}$$

对数曲线在低灰度区陡峭、高灰度区平缓，因此**扩展暗部、压缩亮部**。适用于整体偏暗、需要看清暗部细节的图像（如 X 光片、夜间照片）。

```python
c = 255.0 / np.log(1 + gray.max())
log_img = (c * np.log(1 + gray.astype(np.float64))).astype(np.uint8)
```

### 1.3 伽马校正 (Gamma / Power-Law)

$$s = c \cdot r^\gamma$$

| $\gamma$ 值 | 效果 | 曲线形态 |
|-------------|------|----------|
| $\gamma < 1$ | 整体变亮，暗部细节提升 | 上凸 |
| $\gamma = 1$ | 恒等变换 | 直线 |
| $\gamma > 1$ | 整体变暗，亮部细节提升 | 下凸 |

```python
gamma = 0.5  # γ < 1：变亮
gamma_img = (np.power(gray / 255.0, gamma) * 255).astype(np.uint8)
```

> 显示器、电视机的"伽马值"通常为 2.2，这正是伽马校正名称的由来。

---

## 2. 直方图均衡化 (Histogram Equalization)

### 2.1 什么是直方图？

直方图统计每个灰度级（0–255）出现的频数。

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
```

- 暗图：直方图集中在左侧
- 亮图：直方图集中在右侧
- 低对比度图：直方图集中在中间窄区域
- 高对比度图：直方图分布广泛

### 2.2 全局直方图均衡化

将直方图"拉伸"到整个 [0, 255] 范围，使像素分布接近均匀分布，提升对比度。

```python
equ = cv2.equalizeHist(gray)
```

**原理**：用累积分布函数 (CDF) 作为映射函数。CDF 陡峭处（像素集中）→ 映射后拉开；CDF 平缓处 → 映射后压缩。

**缺点**：全局操作，可能过度放大噪声区域的对比度。

### 2.3 CLAHE（自适应直方图均衡化）

**Contrast Limited Adaptive Histogram Equalization**

1. 把图像分成 $8 \times 8$ 小格 (tile)
2. 每个小格单独做均衡化
3. **对比度限制** (clipLimit)：裁剪过高的直方图柱子再重新分配，防止过度放大噪声
4. 拼接时做双线性插值，消除格子边界

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
```

> CLAHE 是医学图像、卫星图像等场景的标配预处理步骤。

---

## 3. 平滑滤波（去噪/Smoothing）

平滑 = 用邻域像素的某种"平均"替代当前像素，达到模糊/降噪效果。

### 3.1 滤波器对比

| 滤波器 | 函数 | 核心思想 | 优点 | 缺点 |
|--------|------|----------|------|------|
| **均值** | `cv2.blur()` | 邻域直接平均 | 简单快速 | 模糊边缘严重 |
| **高斯** | `cv2.GaussianBlur()` | 邻域高斯加权（中心权重大） | 比均值自然，σ 可调 | 仍会模糊边缘 |
| **中值** | `cv2.medianBlur()` | 邻域取中位数 | ★ 椒盐噪声克星，保边缘 | 对高斯噪声不如均值 |
| **双边** | `cv2.bilateralFilter()` | 空间+像素值双重高斯加权 | ★ 保边去噪 | 速度较慢 |

### 3.2 卷积核 (Kernel)

滤波的本质是**卷积**：一个小矩阵（核）在图像上滑动，每个位置做加权求和。

```
3×3 均值核          3×3 高斯核 (σ≈0.8)
1/9 1/9 1/9         1/16 2/16 1/16
1/9 1/9 1/9         2/16 4/16 2/16
1/9 1/9 1/9         1/16 2/16 1/16
```

- 核越大 → 平滑效果越强 → 但越慢
- 核边长通常为奇数（3, 5, 7...）

### 3.3 中值滤波的独特优势

中值滤波取邻域的**中位数**而非平均。椒盐噪声的极端值（0 或 255）永远不会被选中作为中位数，因此被彻底消除而不会"污染"邻域。

```python
# 核大小必须是奇数（3, 5, 7...）
median = cv2.medianBlur(noisy_img, 5)
```

### 3.4 双边滤波的原理

$$BF[I]_p = \frac{1}{W_p}\sum_{q\in S} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I_p-I_q|) \cdot I_q$$

- $G_{\sigma_s}$：空间高斯权重——距离越近权重越大
- $G_{\sigma_r}$：像素值高斯权重——颜色越接近权重越大

边缘处颜色差大 → 像素值权重接近 0 → 对侧像素不参与平滑 →**边缘被保留**。

---

## 4. 边缘检测 (Edge Detection)

### 4.1 Sobel 算子（一阶导数）

Sobel 算子用两个 3×3 核分别计算水平和竖直梯度：

```
水平梯度 Gx (检测竖直边缘):     竖直梯度 Gy (检测水平边缘):
[-1  0  +1]                     [-1 -2 -1]
[-2  0  +2]                     [ 0  0  0]
[-1  0  +1]                     [+1 +2 +1]
```

梯度幅值（边缘强度）：$G = \sqrt{G_x^2 + G_y^2}$  
梯度方向：$\theta = \arctan(G_y / G_x)$

```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# 取绝对值转回 uint8
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
# 合并
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
```

### 4.2 Laplacian 算子（二阶导数）

Laplacian 计算二阶导数，对灰度突变响应极强——但也对噪声极敏感。通常**先高斯平滑再做 Laplacian**。

```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
```

### 4.3 Canny 边缘检测（★最常用）

Canny 是一个**多阶段**算法，被认为是边缘检测的"黄金标准"：

```
原始图 → [1.高斯平滑] → [2.计算梯度] → [3.非极大值抑制]
       → [4.双阈值检测] → [5.边缘连接] → 最终边缘图
```

1. **高斯平滑**：降噪
2. **计算梯度**（Sobel）：得到幅值和方向
3. **非极大值抑制 (NMS)**：只保留梯度方向上的局部最大值，细化边缘到单像素宽度
4. **双阈值检测**：
   - 梯度 > `high` → **强边缘**（保留）
   - 梯度 < `low` → 丢弃
   - `low` ≤ 梯度 ≤ `high` → **弱边缘**
5. **边缘连接（滞后跟踪）**：弱边缘仅当与强边缘相连时才保留

```python
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
```

> **推荐阈值比**：`high ≈ 2× low` 或 `high ≈ 3× low`

---

## 5. 形态学操作 (Morphological Operations)

形态学操作作用于**二值图像**的白色前景区域。

### 5.1 结构元素 (Structuring Element)

定义"邻域"的形状和大小：

```python
kernel_rect   = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross  = cv2.getStructuringElement(cv2.MORPH_CROSS,   (5, 5))
```

| 形状 | 适用场景 |
|------|----------|
| 矩形 (RECT) | 通用，适合方形物体 |
| 椭圆 (ELLIPSE) | 更平滑，适合自然物体 |
| 十字 (CROSS) | 检测细线、交点 |

### 5.2 基本操作

| 操作 | 原理 | 效果 | OpenCV |
|------|------|------|--------|
| **腐蚀 (Erode)** | 局部最小值 | 白色区域缩小，消除小噪点 | `cv2.erode()` |
| **膨胀 (Dilate)** | 局部最大值 | 白色区域扩大，填充小孔洞 | `cv2.dilate()` |

### 5.3 复合操作

| 操作 | 定义 | 效果 | 应用 |
|------|------|------|------|
| **开运算 (Open)** | 先腐蚀后膨胀 | 去除小物体、断开窄连接 | 去噪点 |
| **闭运算 (Close)** | 先膨胀后腐蚀 | 填充小孔洞、连接邻近物体 | 补裂缝 |
| **形态学梯度 (Gradient)** | 膨胀 − 腐蚀 | 提取物体边界 | 边缘检测 |
| **顶帽 (Top Hat)** | 原图 − 开运算 | 提取比邻域亮的小结构 | 光照不均校正 |
| **黑帽 (Black Hat)** | 闭运算 − 原图 | 提取比邻域暗的小结构 | 检测暗色斑点 |

```python
opened   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat   = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
```

---

## 6. 轮廓检测 (Contour Detection)

### 6.1 查找轮廓

```python
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
```

| 检索模式 | 说明 |
|----------|------|
| `RETR_EXTERNAL` | 只提取最外层轮廓 |
| `RETR_LIST` | 提取所有轮廓，不建立层级 |
| `RETR_TREE` | 提取所有轮廓，建立完整层级树 |
| `RETR_CCOMP` | 提取所有，只分内外两级 |

| 近似方法 | 说明 |
|----------|------|
| `CHAIN_APPROX_NONE` | 存储所有轮廓点 |
| `CHAIN_APPROX_SIMPLE` | 压缩水平/垂直/对角线段，只留端点 |

### 6.2 轮廓属性

```python
area = cv2.contourArea(cnt)            # 面积
perimeter = cv2.arcLength(cnt, True)    # 周长（True=闭合）
x, y, w, h = cv2.boundingRect(cnt)      # 外接矩形
hull = cv2.convexHull(cnt)              # 凸包
```

### 6.3 绘制轮廓

```python
# 绘制所有轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# 绘制单个轮廓
cv2.drawContours(img, [contours[i]], -1, (0, 0, 255), 3)

# 填充轮廓内部
cv2.drawContours(img, [contours[i]], -1, (255, 0, 0), cv2.FILLED)
```

---

## 7. 典型流水线

一个完整的图像处理流水线通常为：

```
原始图 → 灰度化 → 去噪（高斯/中值） → 边缘检测（Canny）
       → 形态学清理（闭运算） → 轮廓提取 → 后处理（外接框/面积过滤）
```

```python
# 完整流水线示例
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(denoised, 50, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(edges_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
significant = [c for c in contours if cv2.contourArea(c) > 50]
```

---

## 8. 核/模板速查

| 需求 | 推荐滤波器 | 推荐核大小 |
|------|-----------|-----------|
| 通用去噪 | 高斯 `GaussianBlur` | 3×3 或 5×5 |
| 椒盐噪声 | 中值 `medianBlur` | 5×5 |
| 保边去噪 | 双边 `bilateralFilter` | d=9, σC=75, σS=75 |
| 高斯噪声 | 高斯 `GaussianBlur` | 5×5 |
| 形态学去小点 | 开运算 `MORPH_OPEN` | 3×3 椭圆 |
| 形态学补洞 | 闭运算 `MORPH_CLOSE` | 3×3 椭圆 |
| 边缘检测 | Canny (50, 150) | — |
| 预处理→Canny | 高斯 (5×5) + Canny | — |

---

## 9. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Canny 边缘太多/太少 | 双阈值不合适 | 调整 low/high 阈值，推荐 high=2×low |
| 形态学操作效果不明显 | kernel 太小 | 增大 kernel 尺寸或增加 iterations |
| 双边滤波太慢 | 双边滤波计算量大 | 对大图先缩小，处理后放大 |
| `findContours` 报错 | OpenCV 版本差异 | 新版返回 `contours, hierarchy`；旧版返回 `image, contours, hierarchy` |
| Sobel 结果为全黑 | ddepth 溢出 | 使用 `cv2.CV_64F`，然后用 `convertScaleAbs` |
| 轮廓数量太多 | 检测到大量细小轮廓 | 按面积过滤：`[c for c in contours if cv2.contourArea(c) > min_area]` |

---

## 10. 下周预告

第 3 周将进入深度学习：PyTorch 基础、张量操作、自动微分、简单神经网络搭建与训练。
