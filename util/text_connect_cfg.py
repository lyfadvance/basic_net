import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict
__C = edict()
cfg = __C
__C.GPU_ID = 2
__C.DETECT_MODE = "H"
class Config:
    SCALE=600
    MAX_SCALE=1200
    TEXT_PROPOSALS_WIDTH=16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO=0.5
    LINE_MIN_SCORE=0.9
    MAX_HORIZONTAL_GAP=50
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.6
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7


