import os
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
deeplabcut.video_inference_superanimal(
    videos=["/root/MouseMotion/videos/20080321162447.mpg"],
    superanimal_name="superanimal_topviewmouse",
    videotype=".mpg",
    pcutoff=0.1
)
