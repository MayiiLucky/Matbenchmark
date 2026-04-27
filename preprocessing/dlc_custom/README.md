# DeepLabCut 自训练数据集复现流程

> 基于实际操作经验整理，适用于小鼠行为关键点检测任务。

---

## 一、环境配置

### 安装 DLC

```bash
conda create -n DEEPLABCUT python=3.12
conda activate DEEPLABCUT
pip install --pre deeplabcut[gui]
```

### 验证安装

```python
import deeplabcut
print(deeplabcut.__version__)  # 应显示 3.0.0+
```

### 注意事项

- Python 3.12 + PyTorch GPU 版本（CUDA 12.x）
- GUI 需要 napari，确保 `deeplabcut[gui]` 安装完整
- 如遇 wayland 警告可忽略，不影响功能

---

## 二、创建项目

```python
import deeplabcut

config_path = deeplabcut.create_new_project(
    'ProjectName',        # 项目名称
    'your_name',          # 操作者名称
    ['/path/to/video.avi'],  # 视频路径列表
    working_directory='/home/user/projects',
    copy_videos=True,
    multianimal=False
)

print(config_path)
# 输出：/home/user/projects/ProjectName-yourname-YYYY-MM-DD/config.yaml
```

---

## 三、配置关键点

编辑 `config.yaml`：

```bash
nano /path/to/ProjectName-yourname-YYYY-MM-DD/config.yaml
```

修改 `bodyparts`：

```yaml
bodyparts:
- nose
- jaw
- neck
- spine1
- spine2
- spine3
- tailbase
- left_shoulder
- left_wrist
- left_forepaw
- right_shoulder
- right_wrist
- right_forepaw
- left_hip
- left_hindpaw
- right_hip
- right_hindpaw

numframes2extract: 50    # 每个视频提取的帧数
start: 0                 # 提取起始位置（0~1）
stop: 1                  # 提取结束位置（0~1）
```

---

## 四、数据集准备

### 方式A：从视频提取帧

```python
deeplabcut.extract_frames(
    config_path,
    mode='automatic',
    algo='kmeans',     # kmeans 覆盖多样场景，比 uniform 更好
    crop=False,
    userfeedback=False
)
```

### 方式B：直接使用图片帧（推荐）

如果已有图片数据集，直接复制到 `labeled-data/` 目录：

```python
import os, shutil, glob

dst = '/path/to/project/labeled-data/my_frames/'
os.makedirs(dst, exist_ok=True)
for f in glob.glob('/path/to/images/*.jpg'):
    shutil.copy(f, dst)

# 同时在 config.yaml 的 video_sets 里添加对应路径
import yaml
with open(config_path) as f:
    cfg = yaml.safe_load(f)
cfg['video_sets'][dst.rstrip('/')] = {'crop': '0, 1280, 0, 1024'}
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
```

### 多数据集合并

如果有多个数据源，用 `merge_datasets` 合并：

```python
deeplabcut.merge_datasets(config_path)
```

> ⚠️ 合并后迭代版本会升级（iteration-0 → iteration-1），后续操作都在新迭代版本里进行。

---

## 五、标注关键点

```python
deeplabcut.label_frames(config_path)
```

打开 napari 标注界面后：

| 操作 | 说明 |
|------|------|
| `+` 图标 | 添加关键点模式 |
| 鼠标左键 | 在图片上点击标注位置 |
| 底部进度条 | 切换帧 |
| `File → Open Folder` | 切换到其他数据集文件夹 |
| `Ctrl+S` | 保存标注 |

### 标注原则

- 每帧标注所有可见关键点，看不清的帧跳过
- 不要强制标注遮挡或模糊的关键点（会产生 NaN 影响训练）
- 关键点定义要前后一致，同一关键点每帧标注同一部位

### 验证标注结果

```python
import pandas as pd

csv = '/path/to/labeled-data/my_frames/CollectedData_yourname.csv'
df = pd.read_csv(csv, header=[0,1,2], index_col=0)
scorer = df.columns.get_level_values(0)[-1]

for kp in df.columns.get_level_values(1).unique():
    if kp.startswith('Unnamed'):
        continue
    valid = df[scorer][kp]['x'].notna().sum()
    print(f'{kp}: {valid}/{len(df)} 帧有效')
```

---

## 六、创建训练集

```python
deeplabcut.create_training_dataset(config_path)
```

### 常见问题

**IndexError: positional indexers are out-of-bounds**

原因：旧训练集索引损坏。解决：

```python
import shutil, glob

for f in glob.glob('/path/to/project/training-datasets/iteration-*'):
    shutil.rmtree(f)

deeplabcut.create_training_dataset(config_path)
```

**训练集帧数少于预期**

原因：部分数据源未被识别。检查：

```python
import pandas as pd

h5 = glob.glob('/path/to/project/training-datasets/**/CollectedData_*.h5', recursive=True)[0]
df = pd.read_hdf(h5)
print(f'训练集总帧数: {len(df)}')
```

---

## 七、训练模型

```python
deeplabcut.train_network(
    config_path,
    shuffle=1,
    device='cuda:0',    # GPU训练
    save_epochs=5,      # 每5个epoch保存一次
    epochs=200
)
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `shuffle` | 1 | 数据集划分编号，重新训练用新编号 |
| `device` | auto | `cuda:0` 指定GPU，`cpu` 强制CPU |
| `save_epochs` | 5 | 保存间隔，越小越占磁盘 |
| `epochs` | 200 | 总训练轮数 |

### 训练过程监控

```
Epoch 200/200 (lr=1e-05), train loss 0.00137, valid loss 0.00743
Model performance:
  metrics/test.rmse:         75.35
  metrics/test.mAP:          97.52
  metrics/test.mAR:          97.50
```

| 指标 | 说明 | 目标值 |
|------|------|--------|
| train loss | 训练损失，越低越好 | < 0.005 |
| test.rmse | 关键点定位误差（像素） | < 10px |
| test.mAP | 平均精度，越高越好 | > 90% |

> ⚠️ 如果 test.mAP 显示 nan 或 0，通常是测试集太少（< 5帧）导致，不代表模型无效，用 `evaluate_network` 进一步验证。

---

## 八、评估模型

```python
deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)
```

### 查看可用的 snapshot

```python
import glob

snapshots = sorted(glob.glob('/path/to/project/dlc-models-pytorch/**/train/snapshot-*.pt', recursive=True))
for i, s in enumerate(snapshots):
    print(f'[{i}] {s}')
```

### 指定 snapshot 评估

在 `config.yaml` 里修改 `snapshotindex`：

```yaml
snapshotindex: 4   # 使用第5个snapshot（从0开始数）
```

---

## 九、视频推理

```python
deeplabcut.analyze_videos(
    config_path,
    ['/path/to/video.mp4'],
    save_as_csv=True,
    snapshot_index=4    # 指定使用哪个snapshot
)
```

### 生成标注视频

```python
deeplabcut.create_labeled_video(
    config_path,
    ['/path/to/video.mp4'],
    draw_skeleton=True,
    snapshot_index=4
)
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `*DLC_*.h5` | 关键点坐标（HDF5） |
| `*DLC_*.csv` | 关键点坐标（CSV） |
| `*DLC_*_labeled.mp4` | 标注后视频 |
| `*DLC_*_meta.pickle` | 元数据 |

---

## 十、常见问题汇总

### Q1:mAP 一直为 0 或 nan

**可能原因**：
1. 测试集太少（< 5帧）→ 增加标注数据
2. 图像质量差（强光、鱼眼畸变）→ 改善采集条件
3. 关键点定义不一致 → 重新标注
4. 旧训练集索引损坏 → 删除重新生成

---

## 十一、数据采集建议

| 项目 | 建议 |
|------|------|
| 图像质量 | 均匀光照，避免强光过曝和反射 |
| 背景 | 目标与背景颜色对比明显 |
| 镜头 | 避免鱼眼镜头，使用普通广角 |
| 标注数量 | 单关键点：100帧以上；多关键点：200帧以上 |
| 数据集划分 | 训练70% / 测试15% / 验证15% |
| 覆盖多样性 | 不同姿态、不同位置均匀覆盖 |
