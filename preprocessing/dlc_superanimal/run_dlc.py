import os
import sys
import glob
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"找到 {len(gpus)} 个 GPU: {[g.name for g in gpus]}")
else:
    print("警告: 未找到 GPU，将使用 CPU 运行")

import deeplabcut

parser = argparse.ArgumentParser(description='DLC SuperAnimal 推理')
parser.add_argument('--video_dir', required=True, help='视频所在目录')
parser.add_argument('--videos', nargs='+', help='指定具体视频文件名，如 A.mp4 B.mp4 C.mp4')
parser.add_argument('--videotype', default='.mp4', help='视频格式 (默认: .mp4)')
parser.add_argument('--pcutoff', type=float, default=0.1, help='置信度阈值 (默认: 0.1)')
parser.add_argument('--model', default='superanimal_topviewmouse', help='SuperAnimal 模型名称')
args = parser.parse_args()

if args.videos:
    videos = [os.path.join(args.video_dir, v) for v in args.videos]
else:
    videos = sorted(glob.glob(os.path.join(args.video_dir, f'*{args.videotype}')))

videos = [v for v in videos if os.path.exists(v)]

if not videos:
    print(f'错误: 没有找到视频文件')
    sys.exit(1)

print(f'找到 {len(videos)} 个视频:')
for v in videos:
    print(f'  {v}')

deeplabcut.video_inference_superanimal(
    videos=videos,
    superanimal_name=args.model,
    videotype=args.videotype,
    pcutoff=args.pcutoff
)
