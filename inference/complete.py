import sys
import time
import logging
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
# ============================================
# CONFIGURATION FOR SCRIPT.PY
# ============================================
MAIN_DIR = r"E:\Projects\hamza Saleem\deepy\Deepshot\CALF"

# Video path (can be from parser)
VIDEO_PATH = r"E:\Projects\hamza Saleem\final2022.mp4"

# Output folder
OUTPUT_FOLDER = os.path.join(MAIN_DIR, "inference", "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# JSON path dynamically based on video filename
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
JSON_PATH = os.path.join(OUTPUT_FOLDER,"Predictions-v2.json")

print("Video path:", VIDEO_PATH)
print("JSON path:", JSON_PATH)

# ============================================
# MAIN PIPELINE
# ============================================
def run_full_pipeline(main_args):
    """
    Runs the complete pipeline:
    1. Runs main.py inference to generate predictions
    2. Runs script.py to extract and create highlights
    """
    
    print("\n" + "="*80)
    print("🚀 STARTING FULL PIPELINE: INFERENCE + HIGHLIGHTS GENERATION")
    print("="*80 + "\n")
    
    # ==========================================
    # STEP 1: RUN MAIN.PY (INFERENCE)
    # ==========================================
    print("📊 STEP 1: Running inference with main.py...\n")
    start_time = time.time()
    
    try:
        # Import and run main from main.py
        sys.path.insert(0, str(Path(__file__).parent))
        from main import main as main_inference
        
        main_inference(main_args)
        
        inference_time = time.time() - start_time
        print(f"\n✅ Inference completed successfully in {inference_time:.2f}s\n")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}\n")
        return False
    
    # ==========================================
    # STEP 2: RUN SCRIPT.PY (HIGHLIGHTS)
    # ==========================================
    print("="*80)
    print("🎬 STEP 2: Generating highlights with script.py...\n")
    start_time = time.time()
    
    try:
        # Import functions from script.py
        sys.path.insert(0, str(Path(__file__).parent))
        from script import (
            load_and_filter_predictions,
            print_analysis,
            extract_clips_fast,
            create_highlights_reel,
            OUTPUT_DIR
        )
        
        # Run the highlights pipeline
        predictions = load_and_filter_predictions(JSON_PATH)
        
        if not predictions:
            print("\n⚠️  No high-quality highlights detected!")
            return False
        
        print_analysis(predictions)
        clips = extract_clips_fast(predictions, VIDEO_PATH, OUTPUT_DIR)
        
        if clips:
            create_highlights_reel(clips)
        
        highlights_time = time.time() - start_time
        print(f"\n✅ Highlights generation completed in {highlights_time:.2f}s\n")
        
    except Exception as e:
        print(f"\n❌ Highlights generation failed: {e}\n")
        return False
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("="*80)
    print("✨ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📈 Total execution time: {time.time() - start_time:.2f}s")
    print(f"📁 Highlights saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    
    # Parse arguments for main.py
    parser = ArgumentParser(
        description='Combined Inference + Highlights Pipeline',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--video_path', required=False, type=str,default="E:\\Projects\\hamza Saleem\\final2022.mp4",
                        help='Path to the SoccerNet-V2 dataset folder')
    
    # Optional arguments for inference
    parser.add_argument('--features', required=False, type=str,
                        default="ResNET_PCA512.npy", help='Video features')
    parser.add_argument('--max_epochs', required=False, type=int,
                        default=1000, help='Maximum number of epochs')
    parser.add_argument('--load_weights', required=False, type=str,
                        default=None, help='Weights to load')
    parser.add_argument('--model_name', required=False, type=str,
                        default="CALF", help='Name of the model to save')
    parser.add_argument('--test_only', required=False, action='store_true',
                        help='Perform testing only')
    parser.add_argument('--challenge', required=False, action='store_true',
                        help='Perform evaluations on the challenge set')
    
    # Model parameters
    parser.add_argument('--num_features', required=False, type=int,
                        default=512, help='Number of input features')
    parser.add_argument('--chunks_per_epoch', required=False, type=int,
                        default=18000, help='Number of chunks per epoch')
    parser.add_argument('--evaluation_frequency', required=False, type=int,
                        default=20, help='Evaluation frequency')
    parser.add_argument('--dim_capsule', required=False, type=int,
                        default=16, help='Dimension of the capsule network')
    parser.add_argument('--framerate', required=False, type=int,
                        default=2, help='Framerate of the input features')
    parser.add_argument('--chunk_size', required=False, type=int,
                        default=120, help='Size of the chunk (in seconds)')
    parser.add_argument('--receptive_field', required=False, type=int,
                        default=40, help='Temporal receptive field (in seconds)')
    
    # Loss weights
    parser.add_argument('--lambda_coord', required=False, type=float,
                        default=5.0, help='Weight of coordinates in detection loss')
    parser.add_argument('--lambda_noobj', required=False, type=float,
                        default=0.5, help='Weight of no object detection in loss')
    parser.add_argument('--loss_weight_segmentation', required=False, type=float,
                        default=0.000367, help='Weight of segmentation loss')
    parser.add_argument('--loss_weight_detection', required=False, type=float,
                        default=1.0, help='Weight of detection loss')
    
    # Training parameters
    parser.add_argument('--batch_size', required=False, type=int,
                        default=32, help='Batch size')
    parser.add_argument('--LR', required=False, type=float,
                        default=1e-03, help='Learning Rate')
    parser.add_argument('--patience', required=False, type=int,
                        default=25, help='Patience before reducing LR')
    
    # System parameters
    parser.add_argument('--GPU', required=False, type=int,
                        default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int,
                        default=4, help='Number of workers to load data')
    parser.add_argument('--loglevel', required=False, type=str,
                        default='INFO', help='Logging level')
    
    # Highlights parameters
    parser.add_argument('--max_total_clips', required=False, type=int,
                        default=20, help='Maximum total clips for highlights')
    parser.add_argument('--before_sec', required=False, type=int,
                        default=7, help='Seconds before event')
    parser.add_argument('--after_sec', required=False, type=int,
                        default=8, help='Seconds after event')
    
    args = parser.parse_args()
    
    # Run the combined pipeline
    success = run_full_pipeline(args)
    
    sys.exit(0 if success else 1)