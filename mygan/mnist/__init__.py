import os

from mygan import TmpFilePath

BATCH_SIZE = 100
NOISE_DIM = 100
OUTPUT_DIR_PATH = os.path.join(TmpFilePath, "mnist")
CHECKPOINT_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, "checkpoint")
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR_PATH, "ckpt")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "log")
CHECKPOINT_INTERVAL = 2

for p in [OUTPUT_DIR_PATH, CHECKPOINT_DIR_PATH]:
    if not os.path.exists(p):
        os.makedirs(p)
