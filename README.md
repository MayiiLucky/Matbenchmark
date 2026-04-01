# 🐭 Keypoint-Based Behavior Analysis

> 记录关键点数据处理结果和行为分析方法的复现流程与代码。

---

## 📁 仓库结构

```
.
├── datasets/                          # 数据集
│   └── mouse_20080321/
│       ├── raw_video/                 # 原始视频
│       ├── labeled_video/             # DLC 标注后视频
│       ├── keypoints/
│       │   ├── keypoints.csv          # 关键点坐标
│       │   └── confidence.csv         # 置信度
│       └── README.md
│
├── benchmarks/                        # 方法复现
│   ├── motionmapper/
│   │   ├── scripts/                   # 核心脚本
│   │   ├── environment/               # 环境配置
│   │   ├── results/
│   │   │   └── mouse_20080321/
│   │   │       ├── behavior_map.png
│   │   │       ├── pcaModes.mat
│   │   │       ├── umap_embedding.mat
│   │   │       └── clusters.mat
│   │   └── README.md
│   ├── vame/
│   └── b-soid/
│
├── utils/                             # 通用工具
└── docs/                              # 文档
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
