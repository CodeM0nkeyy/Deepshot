
import sys
sys.path.insert(0, r'/Users/codemonkey/Desktop/deepy/Deepshot/CALF')

from argparse import Namespace
from inference.complete import run_full_pipeline

args = Namespace(
    video_path=r'/Users/codemonkey/Downloads/final2022.mp4',
    features='ResNET_PCA512.npy',
    max_epochs=1000,
    load_weights=None,
    model_name='CALF',
    test_only=True,
    challenge=False,
    num_features=512,
    chunks_per_epoch=18000,
    evaluation_frequency=20,
    dim_capsule=16,
    framerate=2,
    chunk_size=120,
    receptive_field=40,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    loss_weight_segmentation=0.000367,
    loss_weight_detection=1.0,
    batch_size=32,
    LR=1e-03,
    patience=25,
    GPU=-1,
    max_num_worker=4,
    loglevel='INFO',
    max_total_clips=20,
    before_sec=7,
    after_sec=8
)

success = run_full_pipeline(args)
sys.exit(0 if success else 1)
