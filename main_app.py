import streamlit as st
import sys
import time
import os
from pathlib import Path
import json
import threading
import logging
import subprocess

# Configure logging to file and console
log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "deepshot.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to CLI/console
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DeepShot AI - Football Highlights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .tagline {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
        margin: 1rem 0;
    }
    .contact-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e3c72;
    }
    </style>
""", unsafe_allow_html=True)

def detect_device():
    """Detect available GPU/CPU/AMD device"""
    try:
        import torch
        
        logger.info("=" * 80)
        logger.info("🔍 DEVICE DETECTION")
        logger.info("=" * 80)
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA compiled: {torch.backends.cuda.is_built()}")
        
        # Check for CUDA
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA Available: {device_count} GPU(s) found")
            logger.info(f"   GPU Name: {device_name}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            st.session_state.device = device
            st.session_state.gpu_id = 0
            return device, f"CUDA - {device_name}"
        
        # Check for ROCm (AMD GPUs)
        elif hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            device = "cuda"  # ROCm uses CUDA interface
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AMD GPU"
            logger.info(f"✅ AMD GPU (ROCm) Available: {device_name}")
            logger.info(f"   HIP Version: {torch.version.hip}")
            st.session_state.device = device
            st.session_state.gpu_id = 0
            return device, f"AMD GPU (ROCm) - {device_name}"
        
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("✅ Apple Metal Performance Shaders (MPS) Available")
            st.session_state.device = device
            st.session_state.gpu_id = -1
            return device, "Apple MPS"
        
        # Fallback to CPU
        else:
            device = "cpu"
            logger.info("⚠️  No GPU detected or CUDA not compiled. Using CPU (slower processing)")
            logger.info("💡 For GPU support, rebuild Docker image with CUDA-enabled PyTorch")
            st.session_state.device = device
            st.session_state.gpu_id = -1
            return device, "CPU"
    
    except Exception as e:
        logger.error(f"❌ Error detecting device: {str(e)}")
        logger.info("Defaulting to CPU")
        st.session_state.device = "cpu"
        st.session_state.gpu_id = -1
        return "cpu", "CPU (Error detected)"

# Initialize device detection on startup
if 'device' not in st.session_state:
    device, device_name = detect_device()
else:
    device = st.session_state.device
    device_name = "Previously detected"

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "⚽ Inference", "📧 Contact Us"])

# Show device info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ System Info")
st.sidebar.info(f"**Device:** {device_name}\n\n**Device Type:** {device.upper()}")
st.sidebar.markdown(f"**Log File:** `{log_file}`")

# ====================
# HOME PAGE
# ====================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">⚽ DeepShot AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Transforming Football Matches into Instant Highlights</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=800&h=400&fit=crop", 
                 width="stretch")
    
    st.markdown("---")
    
    st.markdown("## 🌟 What We Offer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI-Powered Analysis</h3>
            <p>Advanced deep learning models detect key moments in football matches with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>⚡ Lightning Fast</h3>
            <p>Process entire matches in minutes and get professional-quality highlights instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🎬 Auto Highlights</h3>
            <p>Automatically creates a highlight reel with the most exciting moments from your match.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🔧 How It Works")
    
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Select your football match video")
    
    with step_col2:
        st.markdown("### 2️⃣ Process")
        st.write("Our AI analyzes every frame")
    
    with step_col3:
        st.markdown("### 3️⃣ Download")
        st.write("Get your highlights reel")
    
    st.markdown("---")
    
    st.markdown("## 🚀 Ready to Get Started?")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Creating Highlights Now ➡️", key="cta_button"):
            st.session_state.page = "⚽ Inference"
            st.rerun()

# ====================
# INFERENCE PAGE
# ====================
elif page == "⚽ Inference":
    st.markdown('<h1 class="main-header">⚽ Video Inference</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Upload your football match and let AI create the highlights</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_clips = st.slider("Maximum clips in highlight reel", 5, 50, 20)
            before_sec = st.slider("Seconds before event", 3, 15, 7)
            after_sec = st.slider("Seconds after event", 3, 15, 8)
            chunk_size = st.slider("Chunk size (seconds)", 60, 240, 120)
        
        with col2:
            batch_size = st.number_input("Batch size", 1, 64, 32)
            framerate = st.number_input("Framerate", 1, 10, 2)
            num_features = st.number_input("Number of features", 128, 1024, 512)
            receptive_field = st.slider("Receptive field (seconds)", 20, 80, 40)
    
    st.markdown("---")
    
    # Initialize session state for video path
    if 'selected_video_path' not in st.session_state:
        st.session_state.selected_video_path = None
    
    st.markdown("### 📤 Upload Video")
    os.makedirs("temp_uploads", exist_ok=True)
    
    upload_tab1, upload_tab2 = st.tabs(["📁 Upload File", "📂 Select from Disk"])
    
    with upload_tab1:
        st.info("⚠️ For files larger than 200MB, please use the 'Select from Disk' tab or configure maxUploadSize in .streamlit/config.toml")
        uploaded_file = st.file_uploader(
            "Choose a football match video (MP4, AVI, MOV)", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your football match video file (recommended for files under 200MB)"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            file_size = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 File size: {file_size:.2f} MB")
            
            st.write("💾 Saving video file...")
            save_progress = st.progress(0)
            
            temp_video_path = os.path.join("temp_uploads", uploaded_file.name)
            os.makedirs("temp_uploads", exist_ok=True)
            
            chunk_size_mb = 1024 * 1024
            bytes_written = 0
            total_size = uploaded_file.size
            
            with open(temp_video_path, "wb") as f:
                while True:
                    chunk = uploaded_file.read(chunk_size_mb)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    save_progress.progress(min(bytes_written / total_size, 1.0))
            
            save_progress.empty()
            st.success("✅ Video saved successfully!")
            
            # Save to session state
            st.session_state.selected_video_path = temp_video_path
    
    with upload_tab2:
        st.info("💡 Select a video file from your repository or enter a path")
        
        # Check if default video exists in repo
        default_video = "final2022.mp4"
        video_exists = os.path.exists(default_video)
        
        if video_exists:
            st.success(f"✅ Default video found: {default_video}")
            
            if st.button("📁 Use Repository Video", key="use_default_video"):
                st.session_state.selected_video_path = default_video
                st.rerun()
            
            st.markdown("---")
            st.write("**Or enter a custom path:**")
        
        file_path_input = st.text_input(
            "Video File Path",
            placeholder="E:\\Projects\\hamza Saleem\\final2022.mp4",
            help="Enter the complete path to your video file"
        )
        
        if file_path_input:
            if os.path.exists(file_path_input):
                # Save to session state
                st.session_state.selected_video_path = file_path_input
                st.success(f"✅ Video found: {os.path.basename(file_path_input)}")
                
                file_size = os.path.getsize(file_path_input) / (1024 * 1024)
                file_size_gb = file_size / 1024
                
                if file_size_gb >= 1:
                    st.info(f"📊 File size: {file_size_gb:.2f} GB")
                else:
                    st.info(f"📊 File size: {file_size:.2f} MB")
            else:
                st.error(f"❌ File not found: {file_path_input}")
                st.session_state.selected_video_path = None
    
    # Use session state for processing
    if st.session_state.selected_video_path:
        temp_video_path = st.session_state.selected_video_path
        
        file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
        
        st.markdown("### 🎥 Video Preview")
        if file_size_mb < 500:
            st.video(temp_video_path)
        else:
            st.warning("⚠️ Video preview disabled for files larger than 500MB to save memory")
            st.info(f"📹 Video ready for processing: {os.path.basename(temp_video_path)}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Processing", key="process_button", type="primary"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.expander("📋 Processing Log", expanded=True)
                
                try:
                    status_text.text("⏳ Initializing pipeline...")
                    progress_bar.progress(10)
                    
                    project_root = Path(__file__).parent.absolute()
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    
                    logger.info(f"Project root: {project_root}")
                    log_container.text(f"✓ Project root: {project_root}")
                    
                    # Determine GPU device ID (-1 for CPU, 0+ for GPU)
                    gpu_device = 0 if st.session_state.device != "cpu" else -1
                    logger.info(f"Using device: {st.session_state.device} (GPU ID: {gpu_device})")
                    
                    script_content = f"""
import sys
import logging
sys.path.insert(0, r'{project_root}')

# Configure logging for subprocess
log_file = r'{log_file}'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from argparse import Namespace
from inference.complete import run_full_pipeline

logger.info("=" * 80)
logger.info("🚀 STARTING FULL PIPELINE: INFERENCE + HIGHLIGHTS GENERATION")
logger.info("=" * 80)

args = Namespace(
    video_path=r'{temp_video_path}',
    features='ResNET_PCA512.npy',
    max_epochs=1000,
    load_weights=None,
    model_name='CALF',
    test_only=True,
    challenge=False,
    num_features={num_features},
    chunks_per_epoch=18000,
    evaluation_frequency=20,
    dim_capsule=16,
    framerate={framerate},
    chunk_size={chunk_size},
    receptive_field={receptive_field},
    lambda_coord=5.0,
    lambda_noobj=0.5,
    loss_weight_segmentation=0.000367,
    loss_weight_detection=1.0,
    batch_size={batch_size},
    LR=1e-03,
    patience=25,
    GPU={gpu_device},
    max_num_worker=4,
    loglevel='INFO',
    max_total_clips={max_clips},
    before_sec={before_sec},
    after_sec={after_sec}
)


logger.info("=" * 80)

try:
    success = run_full_pipeline(args)
    logger.info("✅ Pipeline completed successfully!" if success else "❌ Pipeline failed!")
    sys.exit(0 if success else 1)
except Exception as e:
    logger.error(f"❌ Pipeline error: {{str(e)}}", exc_info=True)
    sys.exit(1)
"""
                    
                    temp_script = os.path.join("temp_uploads", "run_pipeline.py")
                    with open(temp_script, "w") as f:
                        f.write(script_content)
                    
                    logger.info(f"Created temporary script: {temp_script}")
                    log_container.text("✓ Created temporary run script")
                    
                    progress_bar.progress(20)
                    
                    status_text.text("🔍 Running AI inference...")
                    logger.info("📊 STEP 1: Running inference with main.py...")
                    log_container.text("🔍 Step 1: Running inference...")
                    progress_bar.progress(30)
                    
                    try:
                        process = subprocess.Popen(
                            [sys.executable, temp_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            cwd=str(project_root)
                        )
                        
                        output_text = ""
                        output_display = log_container.empty()
                        
                        logger.info("Starting subprocess output capture...")
                        
                        for line in process.stdout:
                            output_text += line
                            logger.info(f"[SUBPROCESS] {line.rstrip()}")
                            
                            if "STEP 1" in line or "inference" in line.lower():
                                progress_bar.progress(40)
                            elif "completed" in line.lower() and "inference" in line.lower():
                                progress_bar.progress(60)
                            elif "STEP 2" in line or "highlights" in line.lower():
                                progress_bar.progress(70)
                            elif "completed" in line.lower() and "highlights" in line.lower():
                                progress_bar.progress(90)
                            
                            display_text = output_text[-2000:] if len(output_text) > 2000 else output_text
                            output_display.text_area("Output", display_text, height=300)
                        
                        return_code = process.wait()
                        
                        logger.info(f"Process completed with return code: {return_code}")
                        output_display.text_area("Full Output", output_text, height=400)
                        
                        if return_code == 0:
                            progress_bar.progress(100)
                            status_text.text("✅ Processing completed successfully!")
                            st.success("🎉 Your highlights are ready!")
                            logger.info("✅ Processing completed successfully!")
                            
                            output_dir = None
                            for line in output_text.split('\n'):
                                if 'Highlights saved to:' in line:
                                    output_dir = line.split('Highlights saved to:')[-1].strip()
                                    break
                            
                            if not output_dir or not os.path.exists(output_dir):
                                possible_dirs = [
                                    os.path.join("highlights_filtered", "match4"),
                                    os.path.join("highlights_filtered"),
                                    os.path.join("inference", "outputs", "highlights"),
                                    os.path.join("outputs", "highlights"),
                                    "highlights"
                                ]
                                
                                for possible_dir in possible_dirs:
                                    if os.path.exists(possible_dir):
                                        output_dir = possible_dir
                                        logger.info(f"Found output directory: {output_dir}")
                                        break
                            
                            if output_dir and os.path.exists(output_dir):
                                st.info(f"📁 Highlights saved to: {output_dir}")
                                logger.info(f"Output directory: {output_dir}")
                                
                                video_files = []
                                for root, dirs, files in os.walk(output_dir):
                                    for file in files:
                                        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                            video_files.append(os.path.join(root, file))
                                
                                if video_files:
                                    st.markdown("### 📥 Generated Files:")
                                    st.success(f"Found {len(video_files)} video file(s)")
                                    logger.info(f"Found {len(video_files)} output video file(s)")
                                    
                                    for file_path in video_files:
                                        file_name = os.path.basename(file_path)
                                        
                                        col_file1, col_file2 = st.columns([3, 1])
                                        
                                        with col_file1:
                                            st.markdown(f"**🎬 {file_name}**")
                                            file_stat = os.stat(file_path)
                                            file_mb = file_stat.st_size / (1024 * 1024)
                                            st.caption(f"Size: {file_mb:.2f} MB | Path: {file_path}")
                                            logger.info(f"Output file: {file_name} ({file_mb:.2f} MB)")
                                        
                                        with col_file2:
                                            with open(file_path, "rb") as f:
                                                st.download_button(
                                                    label="⬇️ Download",
                                                    data=f,
                                                    file_name=file_name,
                                                    mime="video/mp4",
                                                    key=f"download_{file_name}_{hash(file_path)}"
                                                )
                                        
                                        with st.expander(f"👁️ Preview: {file_name}"):
                                            try:
                                                st.video(file_path)
                                            except Exception as e:
                                                st.error(f"Could not load video preview: {e}")
                                                logger.error(f"Video preview error: {str(e)}")
                                        
                                        st.markdown("---")
                                else:
                                    st.warning(f"No video files found in: {output_dir}")
                                    logger.warning(f"No output video files found in: {output_dir}")
                            else:
                                st.warning("⚠️ Could not locate output directory. Check the processing log above for the save location.")
                                logger.warning("Output directory not found")
                        else:
                            progress_bar.empty()
                            status_text.text("❌ Processing failed!")
                            st.error(f"Process exited with code: {return_code}")
                            logger.error(f"Process failed with exit code: {return_code}")
                    
                    except Exception as e:
                        st.error(f"❌ Processing Error: {str(e)}")
                        logger.error(f"Processing error: {str(e)}", exc_info=True)
                        import traceback
                        error_trace = traceback.format_exc()
                        log_container.code(error_trace)
                        progress_bar.empty()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Fatal error: {str(e)}", exc_info=True)
                    import traceback
                    error_trace = traceback.format_exc()
                    
                    with st.expander("🔍 Debug Information", expanded=True):
                        st.code(error_trace)
                        st.code(f"Current directory: {os.getcwd()}")
                        st.code(f"App location: {Path(__file__).parent.absolute()}")
                        
                        inference_dir = Path(__file__).parent / "inference"
                        if inference_dir.exists():
                            st.success(f"✓ Found inference directory: {inference_dir}")
                            files = list(inference_dir.glob("*.py"))
                            st.write("Python files found:")
                            for f in files:
                                st.write(f"  - {f.name}")
                        else:
                            st.error(f"✗ Inference directory not found at: {inference_dir}")
                    
                    progress_bar.empty()

# ====================
# CONTACT PAGE
# ====================
elif page == "📧 Contact Us":
    st.markdown('<h1 class="main-header">📧 Contact Us</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Get in touch with our team</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="contact-card">
            <h3>🏢 Company Information</h3>
            <p><strong>DeepShot AI Technologies</strong></p>
            <p>📍 123 Innovation Drive, Tech Valley<br>
            Silicon City, CA 94000<br>
            United States</p>
            <br>
            <p>📞 Phone: +1 (555) 123-4567</p>
            <p>📧 Email: info@deepshot.ai</p>
            <p>🌐 Website: www.deepshot.ai</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
            <h3>⏰ Business Hours</h3>
            <p><strong>Monday - Friday:</strong> 9:00 AM - 6:00 PM PST</p>
            <p><strong>Saturday:</strong> 10:00 AM - 4:00 PM PST</p>
            <p><strong>Sunday:</strong> Closed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💬 Send us a message")
        
        with st.form("contact_form"):
            name = st.text_input("Your Name *")
            email = st.text_input("Your Email *")
            subject = st.selectbox(
                "Subject *",
                ["General Inquiry", "Technical Support", "Business Partnership", "Feature Request", "Bug Report"]
            )
            message = st.text_area("Message *", height=150)
            
            submitted = st.form_submit_button("📤 Send Message")
            
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you! Your message has been sent successfully. We'll get back to you within 24 hours.")
                    logger.info(f"Contact form submitted: {name} - {subject}")
                else:
                    st.error("❌ Please fill in all required fields.")
        
        st.markdown("---")
        
        st.markdown("""
        <div class="contact-card">
            <h3>🔗 Connect With Us</h3>
            <p>
            <a href="https://twitter.com/deepshot_ai" target="_blank">🐦 Twitter</a> | 
            <a href="https://linkedin.com/company/deepshot-ai" target="_blank">💼 LinkedIn</a> | 
            <a href="https://github.com/deepshot-ai" target="_blank">💻 GitHub</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>© 2024 DeepShot AI. All rights reserved.</p>
        <p>Powered by Deep Learning & Computer Vision</p>
    </div>
""", unsafe_allow_html=True)