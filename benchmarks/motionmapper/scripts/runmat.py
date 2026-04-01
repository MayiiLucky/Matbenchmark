"""
DLC CSV → MotionMapper 完整流程脚本
用法:
    python runmat.py \
        --csv_dir /home/hilda/mousevideo \
        --output_dir /home/hilda/MouseMotion \
        --pattern '*DLC*.csv'
"""

import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import hdf5storage
import h5py
import motionmapperpy as mmpy
from datetime import datetime


# ========== 1. 加载 DLC CSV ==========
def load_dlc_csv(csv_path, pcutoff=0.6):
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]

    keypoints = [
        'nose', 'left_ear', 'right_ear', 'neck', 'spine1',
        'tailbase', 'left_forepaw', 'right_forepaw',
        'left_hindpaw', 'right_hindpaw'
    ]

    n_frames = len(df)
    arr = np.zeros((n_frames, len(keypoints), 2))

    for i, kp in enumerate(keypoints):
        x = df[scorer][kp]['x'].values
        y = df[scorer][kp]['y'].values
        likelihood = df[scorer][kp]['likelihood'].values

        mask = likelihood < pcutoff
        x[mask] = np.nan
        y[mask] = np.nan

        arr[:, i, 0] = x
        arr[:, i, 1] = y

    for i in range(len(keypoints)):
        for c in range(2):
            col = arr[:, i, c]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                idx = np.arange(len(col))
                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
                arr[:, i, c] = col

    return arr


# ========== 2. 自我中心转换 ==========
def egocentric_transform(h5):
    centered = h5 - h5[:, [0], :]

    body_vec = h5[:, 5, :] - h5[:, 0, :]
    angles = np.arctan2(body_vec[:, 1], body_vec[:, 0])

    cos_a = np.cos(-angles - np.pi / 2)
    sin_a = np.sin(-angles - np.pi / 2)

    rotated = np.zeros_like(centered)
    rotated[:, :, 0] = (centered[:, :, 0] * cos_a[:, None]
                        - centered[:, :, 1] * sin_a[:, None])
    rotated[:, :, 1] = (centered[:, :, 0] * sin_a[:, None]
                        + centered[:, :, 1] * cos_a[:, None])

    return rotated


# ========== 3. 体型归一化 ==========
def normalize_by_length(h5_list):
    lengths = [
        np.nanmedian(np.linalg.norm(h5[:, 5] - h5[:, 0], axis=1))
        for h5 in h5_list
    ]
    max_len = np.max(lengths)
    print('  各数据集体型长度: {}'.format(['{:.1f}'.format(l) for l in lengths]))
    return [h5 / (length / max_len) for h5, length in zip(h5_list, lengths)]


# ========== 4. 保存投影并运行 MotionMapper ==========
def run_motionmapper(projections_list, dataset_names, project_path,
                     method='UMAP', fps=30):
    mmpy.createProjectDirectory(project_path)

    for projs, name in zip(projections_list, dataset_names):
        out_path = '{}/Projections/{}_pcaModes.mat'.format(project_path, name)
        hdf5storage.savemat(out_path, {'projections': projs})
        print('  保存投影: {} shape={}'.format(name, projs.shape))

    # 设置参数
    params = mmpy.setRunParameters()
    params.projectPath = project_path
    params.method = method
    params.samplingFreq = fps
    params.minF = 1
    params.maxF = fps // 2
    params.numPeriods = 25
    params.pcaModes = projections_list[0].shape[1]
    params.numProjections = params.pcaModes
    n_files = len(projections_list)
    total_frames = sum(len(p) for p in projections_list)
    min_frames = min(len(p) for p in projections_list)
    params.trainingSetSize = min(5000, total_frames // 4)
    params.training_numPoints = min(
        max(params.trainingSetSize, params.trainingSetSize // n_files + 1),
        min_frames
    )
    params.embedding_batchSize = 30000
    params.useGPU = -1
    params.numProcessors = -1
    params.waveletDecomp = True

    print('  trainingSetSize={}, training_numPoints={}, min_frames={}'.format(
        params.trainingSetSize, params.training_numPoints, min_frames))

    if method == 'UMAP':
        tsnefolder = project_path + '/UMAP/'
        zValstr = 'uVals'
    else:
        tsnefolder = project_path + '/TSNE/'
        zValstr = 'zVals'

    if not os.path.exists(tsnefolder + 'training_embedding.mat'):
        print('\n子采样训练中...')
        mmpy.subsampled_tsne_from_projections(params, project_path)
        print('训练完成！')

    with h5py.File(tsnefolder + 'training_data.mat', 'r') as f:
        trainingSetData = f['trainingSetData'][:].T
    with h5py.File(tsnefolder + 'training_embedding.mat', 'r') as f:
        trainingEmbedding = f['trainingEmbedding'][:].T

    projection_files = sorted(
        glob.glob(project_path + '/Projections/*_pcaModes.mat')
    )
    for i, pfile in enumerate(projection_files):
        out_file = pfile[:-4] + '_{}.mat'.format(zValstr)
        if os.path.exists(out_file):
            print('{}/{}: 已存在，跳过'.format(i + 1, len(projection_files)))
            continue

        print('{}/{}: {}'.format(i + 1, len(projection_files),
                                 os.path.basename(pfile)))
        projections = hdf5storage.loadmat(pfile)['projections']
        zValues, stats = mmpy.findEmbeddings(
            projections, trainingSetData, trainingEmbedding, params
        )
        hdf5storage.write(
            data={'zValues': zValues}, path='/',
            truncate_existing=True, filename=out_file,
            store_python_metadata=False, matlab_compatible=True
        )
        with open(pfile[:-4] + '_{}_stats.pkl'.format(zValstr), 'wb') as f:
            pickle.dump(stats, f)
        del zValues, projections, stats

    print('\nWatershed 分割中...')
    mmpy.findWatershedRegions(
        params, minimum_regions=20, startsigma=1.0,
        pThreshold=[0.33, 0.67], saveplot=True,
        endident='*_pcaModes.mat'
    )
    print('完成！结果保存在: {}'.format(project_path))


# ========== 可视化 ==========
def visualize_results(project_path, output_png, method='UMAP'):
    import matplotlib.pyplot as plt

    wshed_path = '{}/{}/zVals_wShed_groups.mat'.format(project_path, method)
    wshedfile = hdf5storage.loadmat(wshed_path)

    zValues = wshedfile['zValues']
    watershedRegions = wshedfile['watershedRegions'].flatten()
    m = np.abs(zValues).max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _, xx, density = mmpy.findPointDensity(zValues, 1.0, 511, [-m-10, m+10])
    axes[0].imshow(density, cmap=mmpy.gencmap(),
                   extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
    axes[0].set_title('行为密度地图')
    axes[0].axis('off')

    sc = axes[1].scatter(zValues[:, 0], zValues[:, 1],
                         c=watershedRegions, cmap='tab20',
                         s=0.3, alpha=0.4)
    axes[1].set_title('行为分区 ({} 个区域)'.format(int(watershedRegions.max())))
    plt.colorbar(sc, ax=axes[1], label='Region')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print('图像保存至: {}'.format(output_png))


# ========== 主流程 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLC CSV → MotionMapper')
    parser.add_argument('--csv_dir', required=True, help='DLC CSV 文件所在目录')
    parser.add_argument('--output_dir', required=True, help='结果输出目录')
    parser.add_argument('--pattern', default='*DLC*.csv', help='CSV 文件匹配模式')
    parser.add_argument('--method', default='UMAP', choices=['UMAP', 'TSNE'], help='嵌入方法')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--pcutoff', type=float, default=0.6, help='关键点置信度阈值')
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not csv_files:
        print('错误: 没有找到匹配的 CSV 文件')
        exit(1)

    dataset_names = [os.path.basename(f).split('DLC')[0] for f in csv_files]
    print('找到 {} 个文件:'.format(len(csv_files)))
    for i, name in enumerate(dataset_names):
        print('  {}: {}'.format(i + 1, name))

    print('\n步骤 1: 加载 CSV (pcutoff={})...'.format(args.pcutoff))
    h5s = [load_dlc_csv(f, pcutoff=args.pcutoff) for f in csv_files]
    print('  形状: {}'.format(h5s[0].shape))

    print('\n步骤 2: 自我中心转换...')
    ego_h5s = [egocentric_transform(h5) for h5 in h5s]

    print('\n步骤 3: 体型归一化...')
    normed_h5s = normalize_by_length(ego_h5s)

    print('\n步骤 4: 展平关键点坐标...')
    projections_list = [h5.reshape(-1, 20).astype('float32') for h5 in normed_h5s]
    print('  投影形状: {}'.format(projections_list[0].shape))

    print('\n步骤 5: 运行 MotionMapper ({})...'.format(args.method))
    print('开始时间:', datetime.now().strftime('%H:%M:%S'))
    run_motionmapper(projections_list, dataset_names,
                     args.output_dir, method=args.method, fps=args.fps)
    print('结束时间:', datetime.now().strftime('%H:%M:%S'))

    print('\n步骤 6: 生成可视化...')
    png_path = os.path.join(args.output_dir, 'behavior_map.png')
    visualize_results(args.output_dir, png_path, method=args.method)

    print('\n全部完成！')
    print('结果目录:', args.output_dir)
    print('可视化图:', png_path)
