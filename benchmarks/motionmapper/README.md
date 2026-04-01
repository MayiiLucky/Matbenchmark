# MotionMapper 小鼠行为无监督分析复现

基于 [motionmapperpy](https://github.com/bermanlabemory/motionmapperpy) 的小鼠行为无监督聚类分析管道。  
输入 DeepLabCut (DLC) 关键点坐标 CSV，输出行为密度地图与行为簇划分。

---

## 📐 方法架构

```
DLC CSV
  │
  ▼
① 加载关键点坐标
  │  置信度 < pcutoff 的点用线性插值填充
  │
  ▼
② 自我中心转换 (Egocentric Transform)
  │  以 nose 为原点，nose → tailbase 为纵轴旋转对齐
  │  消除全局位移和朝向，只保留姿态信息
  │
  ▼
③ 体型归一化
  │  按 nose → tailbase 中位距离归一化
  │  消除个体体型差异
  │
  ▼
④ 展平为投影向量
  │  10 关键点 × 2 坐标 = 20 维
  │
  ▼
⑤ Morlet 小波变换
  │  捕捉姿态的时频特征
  │  频率范围: 1 ~ fps/2 Hz，25 个频段
  │
  ▼
⑥ UMAP / t-SNE 降维
  │  子采样训练 → 重嵌入全部数据
  │  输出 2D 行为空间坐标
  │
  ▼
⑦ Watershed 分割
     在密度地图上分水岭分割为行为簇
     minimum_regions = 20
```

---

## 📁 项目结构

```
motionmapperpy/
├── motionmapperpy/                  # 核心库（上游原始代码）
│   ├── __init__.py                  # 统一导出接口
│   ├── setrunparameters.py          # 全局参数（频率、UMAP/t-SNE 超参等）
│   ├── motionmapper.py              # 子采样训练 + 新数据重嵌入
│   ├── wavelet.py                   # Morlet 连续小波变换
│   ├── mmutils.py                   # 密度估计、colormap、目录创建
│   ├── wshed.py                     # Watershed 分水岭分区
│   └── demoutils.py                 # 区域视频、转移矩阵等可视化
│
├── runmat.py                        # ★ 主脚本：DLC CSV → MotionMapper 全流程
│
├── demo/                            # 上游 demo
│   ├── demo.py
│   ├── motionmapperpy_fly_demo.ipynb
│   └── motionmapperpy_mouse_demo.ipynb
│
├── data/                            # 示例数据
│   ├── fly/                         # 果蝇 LEAP 关键点 + 视频
│   ├── mice/                        # 小鼠 OFT 视频 + 关键点 npy
│   └── projections/                 # 预计算投影 + 视频
│
├── setup.py                         # pip 安装配置
├── pixi.toml                        # Pixi 环境配置
└── README.md                        # 本文件
```

### 运行输出目录结构

```
output_dir/                          # --output_dir 指定的路径
├── Projections/
│   ├── {name}_pcaModes.mat          # 展平后的投影矩阵
│   ├── {name}_pcaModes_uVals.mat    # UMAP 嵌入坐标
│   └── {name}_pcaModes_uVals_stats.pkl
├── UMAP/
│   ├── training_data.mat            # 子采样训练集
│   ├── training_amps.mat            # 训练集振幅
│   ├── training_embedding.mat       # UMAP 训练嵌入结果
│   ├── umap.model                   # 训练好的 UMAP 模型
│   ├── _trainMeanScale.npy          # 归一化参数
│   ├── zVals_wShed_groups.mat       # Watershed 分区结果
│   └── zWshed*.png                  # 分区可视化图
└── behavior_map.png                 # 行为密度 + 分区可视化
```

---

## 🛠️ 环境配置

### Conda 环境（Python 3.6，已验证可用）

```bash
conda create -n mmenv python=3.6.13 -y
conda activate mmenv

# 按顺序安装（顺序重要）
pip install h5py==2.10.0
pip install numpy scipy scikit-learn==0.24.2 matplotlib
pip install hdf5storage easydict
pip install umap-learn==0.5.1
pip install scikit-image
pip install tqdm pandas
pip install imageio==2.9.0 imageio-ffmpeg==0.4.3 moviepy==1.0.3

cd motionmapperpy
python setup.py install
```

### 验证安装

```bash
python -c "import motionmapperpy as mmpy; print('✓ motionmapperpy OK')"
```

> ⚠️ **GPU 注意**：`cupy`（GPU 加速小波变换）在 CUDA 12.x 下安装可能失败，建议设置 `useGPU = -1` 使用 CPU 计算。

---

## 🚀 复现流程

### 前提条件

- 已有 DeepLabCut 分析输出的 `.csv` 文件（包含 10 个关键点）
- 已激活 `mmenv` 环境

### 运行

```bash
python runmat.py \
    --csv_dir /path/to/csv/files \
    --output_dir /path/to/output \
    --pattern '*DLC*.csv' \
    --method UMAP \
    --fps 30 \
    --pcutoff 0.6
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv_dir` | **必填** | DLC CSV 文件所在目录 |
| `--output_dir` | **必填** | 结果输出目录 |
| `--pattern` | `*DLC*.csv` | CSV 文件名匹配模式 |
| `--method` | `UMAP` | 嵌入方法（`UMAP` / `TSNE`） |
| `--fps` | `30` | 视频帧率 |
| `--pcutoff` | `0.6` | 关键点置信度阈值，低于此值的点被插值替代 |

### 流程详解

| 步骤 | 操作 | 说明 |
|:----:|------|------|
| ① | 加载 DLC CSV | 提取 10 关键点 (x, y)，按 `pcutoff` 过滤并线性插值 |
| ② | 自我中心转换 | nose 为原点、nose → tailbase 为纵轴旋转对齐 |
| ③ | 体型归一化 | 各数据集按 nose–tailbase 中位距离缩放至统一尺度 |
| ④ | 展平投影 | `(N, 10, 2)` → `(N, 20)` float32 投影矩阵 |
| ⑤ | 小波分解 | Morlet 小波变换，捕捉时频特征（25 个频段） |
| ⑥ | UMAP 降维 | 子采样训练集 → 训练 UMAP → 重嵌入全部数据至 2D |
| ⑦ | Watershed 分割 | 在 2D 密度地图上分水岭分割，划分行为簇 |

---

## 📄 参考

- Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014). *Mapping the stereotyped behaviour of freely moving fruit flies.* Journal of The Royal Society Interface.
- 原始 MATLAB 实现：[gordonberman/MotionMapper](https://github.com/gordonberman/MotionMapper)
- Python 移植：[bermanlabemory/motionmapperpy](https://github.com/bermanlabemory/motionmapperpy)
