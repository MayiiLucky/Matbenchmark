# 云服务器 DeepLabCut SuperAnimal 推理指南

> 平台：AutoDL｜GPU：RTX 4090｜Python 3.10

---

## 一、选择镜像（推荐）

在 AutoDL 创建实例时，**直接选对镜像可避免所有 GPU 配置问题**：

| 选项 | 推荐值 |
|------|--------|
| 框架 | TensorFlow 2.10.0 |
| Python | 3.10 |
| CUDA | **11.8**（不要选 12.x） |
| GPU | RTX 4090 / 3090 |

> ⚠️ TensorFlow 2.10 只支持 CUDA 11.x，选 CUDA 12.x 会导致 GPU 无法识别，只能用 CPU（约 2 it/s，68k 帧需要 8+ 小时）。

---

## 二、安装 DeepLabCut

```bash
pip install deeplabcut==2.3.11
```

---

## 三、关键代码

### 运行

**用法：**

```bash
python run_dlc.py --video_dir 视频目录 [选项]
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video_dir` | 视频所在目录（必填） | — |
| `--videos` | 指定具体文件名，多个用空格分隔（可选，不填则处理目录下所有视频） | 全部视频 |
| `--videotype` | 视频格式 | `.mp4` |
| `--pcutoff` | 置信度阈值 | `0.1` |
| `--model` | SuperAnimal 模型 | `superanimal_topviewmouse` |

**示例：**

- 处理目录下所有 mp4：

```bash
python run_dlc.py --video_dir /root/MouseMotion/videos
```

- 指定处理 A、B、C：

```bash
python run_dlc.py --video_dir /root/MouseMotion/videos --videos A.mp4 B.mp4 C.mp4
```

- 处理 mpg 格式：

```bash
python run_dlc.py --video_dir /root/MouseMotion/videos --videotype .mpg
```

- 调整置信度阈值：

```bash
python run_dlc.py --video_dir /root/MouseMotion/videos --videos C.mp4 --pcutoff 0.3
```
---

## 四、输出文件

| 文件 | 内容 |
|------|------|
| `*DLC_snapshot-200000.h5` | 关键点坐标（HDF5 格式） |
| `*DLC_snapshot-200000_labeled.mp4` | 标注后视频 |
| `*DLC_snapshot-200000_meta.pickle` | 元数据 |
| `plot-poses/` | 关键点轨迹图 |

### h5 转 csv

```bash
python -c "
import pandas as pd
df = pd.read_hdf('YOUR_VIDEO_DLC_snapshot-200000.h5')
df.to_csv('YOUR_VIDEO_DLC_snapshot-200000.csv')
print('shape:', df.shape)
print('columns:', df.columns.tolist()[:6])
"
```

---

## 五、性能参考

| 配置 | 速度 | 68k 帧耗时 |
|------|:----:|:---------:|
| CPU only（CUDA 版本不匹配） | ~2 it/s | ~8 小时 |
| RTX 4090 + CUDA 11.8 镜像 | ~29 it/s | ~40 分钟 |

---

## 六、SuperAnimal 关键点（27个）

```
nose, left_ear, right_ear, left_ear_tip, right_ear_tip,
left_eye, right_eye, neck, mid_back, mouse_center,
mid_backend, mid_backend2, mid_backend3, tail_base,
tail1, tail2, tail3, tail4, tail5,
left_shoulder, left_midside, left_hip,
right_shoulder, right_midside, right_hip,
tail_end, head_midpoint
```

---

## 七、故障排除

### CUDA 12.x 镜像修复

如果不得不用 CUDA 12.x 镜像，用以下方法修复：

```bash
# 查找 conda 环境里的 CUDA 11 库
find / -name "libcudart.so.11*" 2>/dev/null

# 设置库路径（通常在 /root/miniconda3/lib/）
export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH
```

然后在脚本开头加一行：

```python
os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
```

### 常见警告（可忽略）

| 警告 | 原因 | 影响 |
|------|------|------|
| `libnvinfer.so.7 not found` | TensorRT 未安装 | 无 |
| `ptxas not found` | TensorFlow 自动用驱动编译 | 无 |
