# 🐭 Keypoint-Based Behavior Analysis

> 记录关键点数据处理结果和行为分析方法的复现流程与代码。

---

## 📁 仓库结构

```
.
Matbenchmark/
│
├── 📄 README.md                    # 主项目文档 (2.5 KB)
│   └── 包含：项目概述、方法对比表、数据集说明、快速开始
│
├── 📁 benchmarks/                  # ★ 核心：行为分析方法复现
│   │
│   ├── motionmapper/               # ✅ MotionMapper (Berman et al. 2014)
│   │   ├── README.md               # 详细复现指南 (262 行，9.7 KB)
│   │   │   └── 内容：方法架构、项目结构、输出文件说明、环境配置、参数详解
│   │   │
│   │   ├── scripts/                # 执行脚本
│   │   │   └── runmat.py           # 主运行脚本 (9.3 KB)
│   │   │       └── 功能：DLC CSV → MotionMapper 完整流程
│   │   │       └── 参数：csv_dir, output_dir, pattern, method, fps, pcutoff
│   │   │
│   │   ├── environment/            # 环境配置
│   │   │   └── dlc_config.yaml     # DeepLabCut 配置文件 (5.5 KB)
│   │   │
│   │   └── results/                # 运行结果
│   │       ├── mouse_20080321/
│   │       │   ├── behavior_map.png          # 行为密度热图 + 分区可视化 (481 KB)
│   │       │   └── zVals_wShed_groups.mat    # 核心结果：2D坐标 + 簇标签 (2.2 MB)
│   │       │
│   │       └── mousetop_20080321/
│   │           └── .gitkeep        # (占位符)
│   │
│   ├── vame/                       # 🚧 VAME (Luxem et al. 2022) [规划中]
│   └── b-soid/                     # 📋 B-SOiD (Hsu & Yttri 2021) [规划中]
│
│
├── 📁 datasets/                    # ★ 数据集：关键点标注数据
│   │
│   ├── mouse_20080321/             # 侧视角小鼠视频数据
│   │   └── keypoints/
│   │       └── keypoints.csv       # 关键点坐标 (20 MB)
│   │           └── 格式：10个关键点 × (x,y)坐标
│   │
│   ├── mousetop_20080321/          # 俯视角小鼠视频数据
│   │   ├── keypoints.csv           # 关键点坐标 (78 MB)
│   │   └── keypoints.h5            # 同数据HDF5格式 (43 MB)
│   │
│   └── writhing_ABC/               # 新增：三只个体的运动数据
│       ├── A/
│       │   ├── keypoints.csv       # 关键点坐标 (2.1 MB)
│       │   └── labeled.mp4         # 标注后视频 (9.9 MB)
│       │
│       ├── B/                      # (空目录)
│       └── C/                      # (空目录)
│
│
├── 📁 preprocessing/               # ★ 数据预处理：关键点提取方法
│   │
│   ├── dlc_custom/                 # 自训练 DeepLabCut 模型
│   │   ├── README.md               # DLC 完整教程 (333 行，7.3 KB)
│   │   │   └── 内容：从项目创建→标注→训练→推理的11步完整指南
│   │   │
│   │   └── .gitkeep                # (占位符)
│   │
│   └── dlc_superanimal/            # SuperAnimal 预训练模型
│       ├── README.md               # SuperAnimal 快速指南 (3.4 KB)
│       ├── pose_cfg.yaml           # 姿态配置文件 (7.2 KB)
│       └── run_dlc.py              # 推理脚本 (1.6 KB)
│
│
└── 📊 统计信息
    ├── 总大小：82.5 MB
    ├── 主要数据文件：3+ GB (LFS)
    ├── 创建时间：32天前
    └── 最后更新：4天前 (2026-04-27)
```

---

## 🎯 已复现方法

| 方法 | 论文 | 核心技术 | 状态 |
|------|------|----------|:----:|
| [MotionMapper](benchmarks/motionmapper/) | Berman et al., 2014 | Wavelet + UMAP + Watershed | ✅ |
| [VAME](benchmarks/vame/) | Luxem et al., 2022 | VAE + RNN | 🚧 |
| [B-SOiD](benchmarks/b-soid/) | Hsu & Yttri, 2021 | Random Forest | 📋 |

---

## 📊 数据集

| 数据集 | 物种 | 帧数 | 采样率 | 行为标注 |
|--------|------|-----:|-------:|----------|
| [mouse_20080321](datasets/mouse_20080321/) | 小鼠 | 68,783 | 30 fps | drink / eat / groom / hang / micromovement / rear / rest / walk |
| 待添加 | — | — | — | — |

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/你的用户名/keypoint-behavior-analysis.git
cd keypoint-behavior-analysis

# 查看某个方法的复现指南
cd benchmarks/motionmapper
cat README.md
```

每个 `benchmarks/` 子目录下都有独立的 README，包含：
- 环境配置步骤
- 完整复现流程
- 结果文件说明

---

## 📈 跨方法结果对比

| 数据集 | 方法 | 行为簇数 | 处理时间 |
|--------|------|:--------:|:--------:|
| mouse_20080321 | MotionMapper | 20 | ~15 min |
| 待添加 | — | — | — |

---

## 📄 License

[MIT License](LICENSE)
