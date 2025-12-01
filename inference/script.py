import json
import subprocess
import os
from pathlib import Path
from collections import defaultdict

# ============================================
# CONFIGURATION
# ============================================
VIDEO_PATH = r"E:\Projects\hamza Saleem\deepy\Benchmarks\CALF\inference\outputs\match4.mp4"
JSON_PATH = r"E:\Projects\hamza Saleem\deepy\Benchmarks\CALF\inference\outputs\Predictions-v2.json"

EVENT_PRIORITY = {
    'Goal': 10,
    'Penalty': 9,
    'Red card': 9,
    'Yellow->red card': 9,
    'Shots on target': 7,
    'Shots off target': 6,
    'Corner': 5,
    'Yellow card': 5,
    'Foul': 4,
    'Offside': 4,
    'Direct free-kick': 4,
    'Clearance': 3,
    'Substitution': 2,
    'Indirect free-kick': 2,
    'Throw-in': 1,
    'Kick-off': 1,
    'Ball out of play': 1
}

CONFIDENCE_THRESHOLDS = {
    'Goal': 0.40,
    'Penalty': 0.40,
    'Red card': 0.50,
    'Yellow->red card': 0.50,
    'Shots on target': 0.65,
    'Shots off target': 0.70,
    'Corner': 0.60,
    'Yellow card': 0.65,
    'Foul': 0.70,
    'Offside': 0.60,
    'Direct free-kick': 0.55,
    'Clearance': 0.75,
    'Substitution': 0.70,
    'Indirect free-kick': 0.75,
    'Throw-in': 0.90,
    'Kick-off': 0.80,
    'Ball out of play': 0.95
}

MAX_CLIPS_PER_TYPE = {
    'Goal': 999,
    'Penalty': 999,
    'Red card': 999,
    'Yellow->red card': 999,
    'Shots on target': 5,
    'Shots off target': 4,
    'Corner': 6,
    'Yellow card': 3,
    'Foul': 5,
    'Offside': 3,
    'Direct free-kick': 4,
    'Clearance': 3,
    'Substitution': 4,
    'Indirect free-kick': 3,
    'Throw-in': 2,
    'Kick-off': 2,
    'Ball out of play': 0
}

MAX_TOTAL_CLIPS = 20
MIN_TIME_BETWEEN_CLIPS = 15

VIDEO_NAME = Path(VIDEO_PATH).stem
OUTPUT_DIR = Path("highlights_filtered") / VIDEO_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# LOAD & FILTER PREDICTIONS
# ============================================
def load_and_filter_predictions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    filtered = []
    for pred in data['predictions']:
        label = pred['label']
        confidence = float(pred['confidence'])
        threshold = CONFIDENCE_THRESHOLDS.get(label, 0.7)
        max_allowed = MAX_CLIPS_PER_TYPE.get(label, 5)

        if confidence >= threshold and max_allowed > 0:
            filtered.append({
                'label': label,
                'time_sec': float(pred['position']) / 1000,
                'confidence': confidence,
                'priority': EVENT_PRIORITY.get(label, 1),
                'half': pred['half']
            })

    print(f"After confidence filter: {len(filtered)} events")
    filtered = remove_close_duplicates(filtered, MIN_TIME_BETWEEN_CLIPS)
    print(f"After removing duplicates: {len(filtered)} events")
    filtered = limit_per_event_type(filtered)
    print(f"After per-type limits: {len(filtered)} events")
    filtered = select_top_events(filtered, MAX_TOTAL_CLIPS)
    print(f"After top-N selection: {len(filtered)} events")
    filtered.sort(key=lambda x: x['time_sec'])
    return filtered

def remove_close_duplicates(predictions, time_window):
    if not predictions:
        return []
    predictions.sort(key=lambda x: x['time_sec'])
    result = []
    i = 0
    while i < len(predictions):
        group = [predictions[i]]
        j = i + 1
        while j < len(predictions) and predictions[j]['time_sec'] - predictions[i]['time_sec'] <= time_window:
            group.append(predictions[j])
            j += 1
        best = max(group, key=lambda x: (x['priority'], x['confidence']))
        result.append(best)
        i = j
    return result

def limit_per_event_type(predictions):
    by_type = defaultdict(list)
    for pred in predictions:
        by_type[pred['label']].append(pred)
    result = []
    for label, events in by_type.items():
        max_allowed = MAX_CLIPS_PER_TYPE.get(label, 5)
        events.sort(key=lambda x: x['confidence'], reverse=True)
        result.extend(events[:max_allowed])
    return result

def select_top_events(predictions, max_total):
    if len(predictions) <= max_total:
        return predictions
    for pred in predictions:
        pred['score'] = pred['priority'] * pred['confidence']
    predictions.sort(key=lambda x: x['score'], reverse=True)
    return predictions[:max_total]

# ============================================
# ANALYSIS
# ============================================
def print_analysis(predictions):
    print("\n" + "="*80)
    print("🎬 HIGHLIGHTS SUMMARY")
    print("="*80)
    by_type = defaultdict(list)
    for pred in predictions:
        by_type[pred['label']].append(pred)
    total_duration = len(predictions) * 15
    print(f"\nTotal clips: {len(predictions)}")
    print(f"Estimated duration: {total_duration//60}m {total_duration%60}s")
    print("\nBreakdown by event type:")
    print(f"{'Event':<25} {'Count':<8} {'Avg Conf':<10}")
    print("-"*80)
    for label in sorted(by_type.keys(), key=lambda x: EVENT_PRIORITY.get(x, 0), reverse=True):
        events = by_type[label]
        avg_conf = sum(e['confidence'] for e in events) / len(events)
        print(f"{label:<25} {len(events):<8} {avg_conf:.3f}")
    print("\n" + "="*80)
    print("📍 TIMELINE")
    print("="*80)
    for i, pred in enumerate(predictions, 1):
        mins = int(pred['time_sec'] // 60)
        secs = int(pred['time_sec'] % 60)
        priority_stars = "⭐" * min(pred['priority'], 5)
        print(f"{i:2d}. {mins:02d}:{secs:02d} - {pred['label']:<25} {priority_stars} (conf: {pred['confidence']:.3f})")

# ============================================
# CLIP EXTRACTION
# ============================================
def create_segments(predictions, before_sec=7, after_sec=8):
    segments = []
    print(f"\n⚡ Preparing {len(predictions)} clips...")
    for i, pred in enumerate(predictions):
        start_time = max(0, pred['time_sec'] - before_sec)
        end_time = pred['time_sec'] + after_sec
        mins = int(pred['time_sec'] // 60)
        secs = int(pred['time_sec'] % 60)
        segments.append({
            'index': i,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'label': pred['label'],
            'time_str': f"{mins:02d}m{secs:02d}s"
        })
        print(f"  ✓ [{i+1}/{len(predictions)}] {pred['label']} @ {mins}:{secs:02d}")
    return segments

def extract_clips_fast(predictions, video_path, output_dir, before_sec=7, after_sec=8):
    if not predictions:
        print("\n⚠️ No clips to extract")
        return []
    segments = create_segments(predictions, before_sec, after_sec)
    clips_info = []
    print(f"\n✂️  Extracting {len(segments)} clips (copy mode)...\n")
    for segment in segments:
        label_safe = segment['label'].replace(' ', '_').replace('->', '_to_')
        clip_file = output_dir / f"clip_{segment['index']:03d}_{label_safe}_{segment['time_str']}.mp4"
        cmd = [
            'ffmpeg',
            '-ss', str(segment['start']),
            '-i', video_path,
            '-t', str(segment['duration']),
            '-c', 'copy',
            '-y',
            '-loglevel', 'error',
            str(clip_file)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips_info.append({'path': str(clip_file), 'prediction': predictions[segment['index']]})
    print(f"✓ Extracted {len(clips_info)} clips!")
    return clips_info

# ============================================
# FINAL MERGE (RE-ENCODED)
# ============================================
def create_highlights_reel(clips_info, output_file=None):
    if not clips_info:
        return None
    if output_file is None:
        output_file = OUTPUT_DIR / f"{VIDEO_NAME}_HIGHLIGHTS.mp4"

    concat_file = OUTPUT_DIR / 'clips_concat.txt'
    with open(concat_file, 'w', encoding='utf-8') as f:
        for clip in clips_info:
            clip_path = os.path.abspath(clip['path'])
            clip_path = clip_path.replace("\\", "/")  # FFmpeg prefers forward slashes
            f.write(f"file '{clip_path}'\n")

    print("\n🎬 Creating final highlights reel (RE-ENCODED)...\n")

    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',
        '-loglevel', 'error',
        str(output_file)
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n❌ FFmpeg merge failed!")
        print(proc.stderr)
        return None

    if not os.path.exists(output_file):
        print("\n❌ Output file was not created!")
        return None

    os.remove(concat_file)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"✅ HIGHLIGHTS CREATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📹 File: {output_file}")
    print(f"📊 Size: {size_mb:.1f} MB")
    print(f"🎬 Clips: {len(clips_info)}")
    print(f"🔊 Audio: Re-encoded (AAC)")
    print(f"🎥 Video: Re-encoded (H.264)")
    print(f"{'='*80}\n")
    return str(output_file)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("⚽ FAST FOOTBALL HIGHLIGHTS GENERATOR")
    print("="*80)
    print(f"Video: {VIDEO_NAME}")
    print(f"Target: {MAX_TOTAL_CLIPS} best moments")
    print(f"Mode: Fast extraction + re-encoded final output")
    print("="*80)

    predictions = load_and_filter_predictions(JSON_PATH)

    if not predictions:
        print("\n❌ No high-quality highlights detected!")
    else:
        print_analysis(predictions)
        clips = extract_clips_fast(predictions, VIDEO_PATH, OUTPUT_DIR)
        if clips:
            create_highlights_reel(clips)
