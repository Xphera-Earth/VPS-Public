# xEarth Visual Positioning System (VPS) by XPHERA

A high-performance visual localization system that determines camera pose from a single image by matching against a pre-built 3D point cloud database. The system uses xEarth's feature extraction and Qdrant vector database for fast, accurate localization.

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** with CUDA support (required for optimal performance)
- **Python 3.12+** 
- **Astral uv** (Python package manager)
- **Qdrant database** (vector database for feature storage)
- **Docker** (for containerized deployment)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd VPS
```

2. **Install dependencies using uv:**
```bash
uv sync
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your Qdrant server details
```

## 📋 System Overview

The VPS system operates in two phases:

1. **Preprocessing**: Extract features from E57 point cloud files and index them in Qdrant
2. **Localization**: Match query images against the database to determine camera pose

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Qdrant Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
QDRANT_PREFER_GRPC=true
QDRANT_COLLECTION_PREFIX=e57

# E57 File Path
E57_FILE=/path/to/your/scan.e57

# Performance Tuning
OBS_BATCH_SIZE=500000
OBS_NUM_WORKERS=2
LOC_QUERY_TOPK=3500
PER_IMAGE_GPU=true
```

## 📊 Usage

### Step 1: Preprocessing Pipeline (One-time setup)

The preprocessing pipeline extracts features from E57 point cloud files and builds a searchable database. This is a **one-time process** that can take several hours for large datasets.

#### Prerequisites Check

Before running preprocessing, verify your system meets the requirements:

```bash
# Check NVIDIA GPU availability
nvidia-smi

# Check CUDA availability in Python
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Qdrant server is running
curl http://localhost:6333/health
```

#### Step 1.1: Configure Environment Variables

Edit your `.env` file with the correct paths and settings:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required Environment Variables:**

```bash
# E57 File Path (REQUIRED)
E57_FILE=/path/to/your/scan.e57

# Qdrant Database Configuration (REQUIRED)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
QDRANT_COLLECTION_PREFIX=e57

# Performance Settings (OPTIONAL)
OBS_BATCH_SIZE=500000              # Batch size for database writes
OBS_NUM_WORKERS=2                  # CPU workers for parallel processing
PREPROCESS_PIXEL_TOLERANCE=3.0     # Feature matching tolerance (pixels)
OBS_LOCAL_INDEX=true               # Enable local caching for speed
OBS_LOCAL_INDEX_DIR=./obs_cache    # Cache directory

# Database Management (OPTIONAL)
QDRANT_RECREATE=false              # Set to 'true' to rebuild collections
```

#### Step 1.2: Verify E57 File

Check your E57 file is valid and accessible:

```bash
# Quick file inspection
uv run python -c "
from e57_utils import get_image_count, get_pointcloud_count
e57_file = '/path/to/your/scan.e57'
print(f'Images: {get_image_count(e57_file)}')
print(f'Point clouds: {get_pointcloud_count(e57_file)}')
"
```

Expected output:
```
Images: 90
Point clouds: 90
```

#### Step 1.3: Start Qdrant Database

If using Docker:
```bash
# Start Qdrant server
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

If using local installation:
```bash
# Start Qdrant service
systemctl start qdrant
# or
qdrant --config-path ./config.yaml
```

#### Step 1.4: Run Preprocessing Pipeline

Execute the main preprocessing script:

```bash
# Run with default settings
uv run main.py
```

**What happens during preprocessing:**

1. **File Validation** (30 seconds)
   - Loads and validates E57 file
   - Counts images and point clouds
   - Tests feature extraction on first image

2. **Feature Extraction & Track Building** (1-4 hours)
   - Extracts features from all images
   - Builds 3D point tracks across multiple views
   - Uses GPU acceleration when available
   - Automatically handles memory management

3. **Database Indexing** (30 minutes - 2 hours)
   - Creates Qdrant collections
   - Indexes features and 3D points
   - Builds search indices for fast retrieval

#### Step 1.5: Monitor Progress

The preprocessing script provides detailed progress information:

```bash
E57 file has 90 pointclouds and 90 images
Detected 2048 keypoints
Computing multiview tracks from E57 geometry with streaming GPU...
Streaming GPU succeeded.
Tracks built for 1,234,567 unique 3D points
Qdrant collections: ['e57_obs', 'e57_points3d', 'e57_images']
Preprocess pipeline completed in 3847.23s
```

#### Step 1.6: Verify Preprocessing Success

Check that collections were created successfully:

```bash
# Check Qdrant collections
curl http://localhost:6333/collections

# Verify collection sizes
curl http://localhost:6333/collections/e57_obs
curl http://localhost:6333/collections/e57_points3d
curl http://localhost:6333/collections/e57_images
```

Expected response should show collections with point counts:
```json
{
  "result": {
    "status": "green",
    "points_count": 1234567,
    "indexed_vectors_count": 1234567
  }
}
```

#### Advanced Configuration Options

##### GPU Memory Management

For systems with limited GPU memory:

```bash
# Reduce initial chunk size
export INITIAL_CHUNK_SIZE_POINTS=1000000

# Force CPU processing
export CUDA_VISIBLE_DEVICES=""
```

##### Processing Specific Images

To process only a subset of images (for testing):

```python
# Modify main.py temporarily
def main():
    load_dotenv()
    e57_file = os.getenv("E57_FILE")
    
    # Process only first 10 images
    image_indices = list(range(10))
    
    # ... rest of processing with image_indices parameter
```

##### Performance Tuning

For faster processing on high-end systems:

```bash
# Increase batch sizes
export OBS_BATCH_SIZE=1000000
export OBS_NUM_WORKERS=4

# Increase flush thresholds
export OBS_FLUSH_THRESHOLD=50000
```

#### Troubleshooting Preprocessing

**Common Issues and Solutions:**

1. **CUDA Out of Memory**
   ```bash
   # Reduce chunk size
   export INITIAL_CHUNK_SIZE_POINTS=500000
   
   # Or force CPU processing
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Qdrant Connection Failed**
   ```bash
   # Check Qdrant is running
   curl http://localhost:6333/health
   
   # Check environment variables
   echo $QDRANT_URL
   ```

3. **E57 File Not Found**
   ```bash
   # Verify file path
   ls -la /path/to/your/scan.e57
   
   # Check permissions
   file /path/to/your/scan.e57
   ```

4. **Slow Processing**
   ```bash
   # Enable local indexing
   export OBS_LOCAL_INDEX=true
   export OBS_LOCAL_INDEX_DIR=./obs_cache
   
   # Increase worker count
   export OBS_NUM_WORKERS=4
   ```

#### Expected Processing Times

| Dataset Size | GPU | Processing Time |
|--------------|-----|-----------------|
| 50 images | RTX 3080 | 30-60 minutes |
| 100 images | RTX 3080 | 1-2 hours |
| 200 images | RTX 3080 | 3-4 hours |
| 50 images | CPU only | 2-4 hours |
| 100 images | CPU only | 6-8 hours |

#### Resuming Interrupted Processing

If preprocessing is interrupted, you can resume by:

```bash
# Set recreate to false to append to existing collections
export QDRANT_RECREATE=false

# Run preprocessing again
uv run main.py
```

The system will skip already processed data and continue from where it left off.

### Step 2: Localization

Localize a query image against the preprocessed database:

```bash
uv run fast_localize_accurate.py \
    path/to/intrinsics.json \
    path/to/query_image.jpg \
    --rotation-correction RX_180 \
    --out result.json
```

**Example with your data:**
```bash
uv run fast_localize_accurate.py \
    ar_captures/20250422/20250422_010331_427485_intrinsics.json \
    ar_captures/20250422/20250422_010321_296868_image.jpg \
    --rotation-correction RX_180 \
    --out test_fast_final.json
```

### Intrinsics JSON Format

The intrinsics file should contain camera parameters:

```json
{
  "fx": 800.0,
  "fy": 800.0,
  "cx": 320.0,
  "cy": 240.0,
  "width": 640,
  "height": 480
}
```

Alternative formats are also supported:
```json
{
  "intrinsics": {
    "focal_length_x": 800.0,
    "focal_length_y": 800.0,
    "principal_point_x": 320.0,
    "principal_point_y": 240.0
  }
}
```

## 🌐 API Server Deployment

### Creating an API Server

Create a Flask/FastAPI server to handle localization requests:

```python
# api_server.py
from flask import Flask, request, jsonify
import tempfile
import os
from PIL import Image
import json
from new_localize_corrected import localize_image_with_config

app = Flask(__name__)

@app.route('/localize', methods=['POST'])
def localize():
    try:
        # Get uploaded files
        image_file = request.files['image']
        intrinsics_data = request.json.get('intrinsics')
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
            image_file.save(tmp_img.name)
            
        # Save intrinsics temporarily
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_intr:
            json.dump(intrinsics_data, tmp_intr)
            
        try:
            # Run localization
            result = localize_image_with_config(
                intrinsics_path=tmp_intr.name,
                image_path=tmp_img.name,
                config_path="fast_accurate_cfg.json"
            )
            
            return jsonify({
                'success': True,
                'pose': result.get('pose', {}),
                'inliers': result.get('inliers', 0),
                'processing_time': result.get('processing_time', 0)
            })
            
        finally:
            # Cleanup temporary files
            os.unlink(tmp_img.name)
            os.unlink(tmp_intr.name)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### API Usage Example

```bash
# Using curl
curl -X POST http://localhost:5000/localize \
  -F "image=@query_image.jpg" \
  -H "Content-Type: application/json" \
  -d '{
    "intrinsics": {
      "fx": 800.0,
      "fy": 800.0,
      "cx": 320.0,
      "cy": 240.0
    }
  }'
```

```python
# Using Python requests
import requests

with open('query_image.jpg', 'rb') as img:
    response = requests.post('http://localhost:5000/localize', 
        files={'image': img},
        json={
            'intrinsics': {
                'fx': 800.0, 'fy': 800.0,
                'cx': 320.0, 'cy': 240.0
            }
        }
    )
    
result = response.json()
print(f"Pose: {result['pose']}")
```

### JSON Response Format

The API returns a comprehensive JSON response with camera pose and localization metadata:

#### Successful Localization Response

```json
{
  "success": true,
  "inlier_count": 27,
  "fx": 1337.87,
  "fy": 1337.87,
  "cx": 719.75,
  "cy": 965.03,
  "image_size": [1440, 1920],
  "pose": {
    "R_cw": [[0.917, -0.382, -0.113], [0.160, 0.094, 0.983], [-0.365, -0.919, 0.147]],
    "t_cw": [-6.757, -2.624, 3.706],
    "R_wc": [[0.917, 0.160, -0.365], [-0.382, 0.094, -0.919], [-0.113, 0.983, 0.147]],
    "t_wc": [7.969, 1.073, 1.271],
    "q_wc_wxyz": [0.735, 0.647, -0.086, -0.184],
    "T_wc": [[0.917, 0.160, -0.365, 7.969], [-0.382, 0.094, -0.919, 1.073], [-0.113, 0.983, 0.147, 1.271], [0.0, 0.0, 0.0, 1.0]]
  },
  "matches": 1199,
  "matched_gids": [8614766, 59667929, ...],
  "image_votes": {"21": 15661, "31": 14856, ...},
  "rotation_correction": "RX_180",
  "processing_time": 2.34
}
```

#### Failed Localization Response

```json
{
  "success": false,
  "inlier_count": 2,
  "message": "No valid pose estimated",
  "error": "Insufficient feature matches"
}
```

### Response Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether localization succeeded |
| `inlier_count` | integer | Number of inlier matches used for pose estimation |
| `fx`, `fy`, `cx`, `cy` | float | Camera intrinsic parameters (pixels) |
| `image_size` | [int, int] | Image dimensions [width, height] |
| `pose.R_cw` | 3x3 matrix | Rotation matrix from world to camera coordinates |
| `pose.t_cw` | [float, float, float] | Translation vector from world to camera (meters) |
| `pose.R_wc` | 3x3 matrix | Rotation matrix from camera to world coordinates |
| `pose.t_wc` | [float, float, float] | Camera position in world coordinates (meters) |
| `pose.q_wc_wxyz` | [float, float, float, float] | Camera orientation as quaternion (w,x,y,z) |
| `pose.T_wc` | 4x4 matrix | Complete transformation matrix (camera to world) |
| `matches` | integer | Total number of feature matches found |
| `matched_gids` | [int, ...] | IDs of matched 3D points in the database |
| `image_votes` | {string: int} | Vote counts per database image |
| `rotation_correction` | string | Applied rotation correction (e.g., "RX_180") |
| `processing_time` | float | Total processing time in seconds |

### Using the Pose Data

The response provides multiple pose representations for different use cases:

#### For AR/VR Applications (Unity, Unreal Engine)
```python
# Extract camera position and rotation
position = result['pose']['t_wc']  # [x, y, z] in meters
quaternion = result['pose']['q_wc_wxyz']  # [w, x, y, z]

# Unity example (C#)
Vector3 cameraPosition = new Vector3(position[0], position[1], position[2]);
Quaternion cameraRotation = new Quaternion(quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
```

#### For Robotics (ROS, OpenCV)
```python
import numpy as np

# Get transformation matrix
T_wc = np.array(result['pose']['T_wc'])

# Extract rotation and translation
R_wc = T_wc[:3, :3]
t_wc = T_wc[:3, 3]

# Convert to ROS geometry_msgs/Pose
from geometry_msgs.msg import Pose, Point, Quaternion
pose_msg = Pose()
pose_msg.position = Point(x=t_wc[0], y=t_wc[1], z=t_wc[2])
q = result['pose']['q_wc_wxyz']
pose_msg.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
```

#### For Computer Vision (OpenCV, Camera Calibration)
```python
# Use camera-to-world matrices for projection
R_cw = np.array(result['pose']['R_cw'])
t_cw = np.array(result['pose']['t_cw'])

# Project 3D world points to image
world_points = np.array([[1.0, 2.0, 3.0]])  # Example 3D point
camera_matrix = np.array([[result['fx'], 0, result['cx']], 
                         [0, result['fy'], result['cy']], 
                         [0, 0, 1]])

# Project to image coordinates
image_points, _ = cv2.projectPoints(world_points, 
                                   cv2.Rodrigues(R_cw)[0], 
                                   t_cw, 
                                   camera_matrix, 
                                   None)
```

### Quality Assessment

Use these fields to assess localization quality:

```python
def assess_localization_quality(result):
    if not result['success']:
        return "FAILED"
    
    inliers = result['inlier_count']
    matches = result['matches']
    
    if inliers >= 30:
        return "EXCELLENT"
    elif inliers >= 15:
        return "GOOD" 
    elif inliers >= 8:
        return "FAIR"
    else:
        return "POOR"

# Usage
quality = assess_localization_quality(result)
print(f"Localization quality: {quality}")
```

### Error Handling

```python
def handle_localization_response(response):
    try:
        result = response.json()
        
        if result['success']:
            # Use pose data
            position = result['pose']['t_wc']
            orientation = result['pose']['q_wc_wxyz']
            confidence = result['inlier_count']
            
            return {
                'position': position,
                'orientation': orientation,
                'confidence': confidence
            }
        else:
            # Handle failure
            print(f"Localization failed: {result.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Error processing response: {e}")
        return None
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    curl \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN uv sync

# Expose API port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Run the API server
CMD ["uv", "run", "api_server.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333

  vps-api:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - qdrant
    environment:
      - QDRANT_URL=http://qdrant:6333
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/app/data
      - ./obs_cache:/app/obs_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  qdrant_data:
```

### Build and Run

```bash
# Build and start services
docker-compose up --build

# Run preprocessing (one-time)
docker-compose exec vps-api uv run main.py

# Test localization
curl -X POST http://localhost:5000/localize \
  -F "image=@test_image.jpg" \
  -H "Content-Type: application/json" \
  -d '{"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}}'
```

## 📁 Multiple E57 Files Support

To process multiple E57 files in a directory as a single space, modify `main.py`:

```python
# Enhanced main.py for multiple E57 files
import glob
from pathlib import Path

def preprocess_multiple_e57_files(e57_directory: str, collection_prefix: str = "multi_e57", recreate: bool = False):
    """Process multiple E57 files in a directory as a unified space."""
    
    e57_files = glob.glob(os.path.join(e57_directory, "*.e57"))
    if not e57_files:
        raise ValueError(f"No E57 files found in {e57_directory}")
    
    print(f"Found {len(e57_files)} E57 files to process")
    
    client = init_qdrant_client()
    all_tracks = {}
    all_image_stats = {}
    point_id_offset = 0
    image_id_offset = 0
    
    for i, e57_file in enumerate(e57_files):
        print(f"\nProcessing E57 file {i+1}/{len(e57_files)}: {e57_file}")
        
        # Compute tracks for this file
        if torch.cuda.is_available():
            tracks_by_point, image_stats = compute_multiview_tracks_from_e57_gpu_streaming(
                e57_file_path=e57_file,
                image_indices=None,
                occlusion_downscale=2,
                initial_chunk_size_points=2_000_000,
            )
        else:
            tracks_by_point, image_stats = compute_multiview_tracks_from_e57(
                e57_file_path=e57_file,
                image_indices=None,
                occlusion_downscale=2,
            )
        
        # Offset point and image IDs to avoid conflicts
        offset_tracks = {}
        for point_id, tracks in tracks_by_point.items():
            new_point_id = point_id + point_id_offset
            offset_tracks[new_point_id] = [
                (img_id + image_id_offset, x, y) for img_id, x, y in tracks
            ]
        
        offset_image_stats = {}
        for img_id, stats in image_stats.items():
            new_img_id = img_id + image_id_offset
            # Add source file information
            stats = stats.copy()
            stats['source_e57'] = e57_file
            stats['original_image_id'] = img_id
            offset_image_stats[new_img_id] = stats
        
        # Merge into global collections
        all_tracks.update(offset_tracks)
        all_image_stats.update(offset_image_stats)
        
        # Update offsets for next file
        point_id_offset += max(tracks_by_point.keys()) + 1 if tracks_by_point else 0
        image_id_offset += max(image_stats.keys()) + 1 if image_stats else 0
    
    print(f"\nCombined: {len(all_tracks)} unique 3D points across {len(all_image_stats)} images")
    
    # Index combined data into Qdrant
    names = index_e57_to_qdrant(
        client=client,
        e57_file_path=e57_directory,  # Pass directory instead of single file
        tracks_by_point=all_tracks,
        image_stats=all_image_stats,
        collection_prefix=collection_prefix,
        recreate_collections=recreate,
        pixel_tolerance=float(os.getenv("PREPROCESS_PIXEL_TOLERANCE", "3.0")),
    )
    
    return names

def main():
    load_dotenv()
    
    # Check if processing single file or directory
    e57_path = os.getenv("E57_FILE", "/home/ion/scans/jgapt/jgapt.e57")
    collection_prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "e57")
    recreate = os.getenv("QDRANT_RECREATE", "false").lower() == "true"
    
    if os.path.isdir(e57_path):
        print(f"Processing multiple E57 files in directory: {e57_path}")
        preprocess_multiple_e57_files(e57_path, collection_prefix, recreate)
    else:
        print(f"Processing single E57 file: {e57_path}")
        preprocess_pipeline(e57_path, collection_prefix, recreate)
```

### Usage for Multiple Files

```bash
# Set directory containing E57 files
export E57_FILE=/path/to/e57/directory/
export QDRANT_COLLECTION_PREFIX=multi_site

# Run preprocessing
uv run main.py
```

## ⚡ Performance Optimization

### GPU Memory Management

The system automatically handles GPU memory:
- Tries streaming GPU processing first
- Falls back to non-streaming GPU if OOM
- Falls back to CPU if no CUDA available

### Configuration Tuning

Key parameters in `fast_accurate_cfg.json`:

```json
{
  "topk_per_desc": 100,        // Features per descriptor (higher = more accurate, slower)
  "batch_size": 512,           // Batch size for processing
  "min_pnp_matches": 4,        // Minimum matches for PnP solver
  "pnp_reproj_error": 10.0,    // Reprojection error threshold
  "top_images_count": 15,      // Number of candidate images
  "per_image_limit": 4         // Matches per image
}
```

### Environment Variables for Speed

```bash
# Query optimization
LOC_QUERY_TOPK=3500          # Reduce keypoints for speed
PER_IMAGE_GPU=true           # Use GPU for per-image matching

# Batch processing
OBS_BATCH_SIZE=512           # Observation batch size
OBS_FLUSH_THRESHOLD=10000    # Flush threshold

# PnP optimization
PNP_LM_MAX_ITERS=30          # Levenberg-Marquardt iterations
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `LOC_QUERY_TOPK`
   - Lower `OBS_BATCH_SIZE`
   - Use CPU fallback

2. **Qdrant Connection Failed**
   - Check `QDRANT_URL` in `.env`
   - Verify Qdrant server is running
   - Check API key if using authentication

3. **Low Localization Accuracy**
   - Increase `topk_per_desc` in config
   - Check camera intrinsics accuracy
   - Verify query image quality

4. **Slow Performance**
   - Enable GPU acceleration
   - Tune batch sizes
   - Use local observation indexing

### Debug Mode

Enable debug output:

```bash
export OBS_DEBUG=true
export LOC_DEBUG_SCALE=true

uv run fast_localize_accurate.py --debug \
    intrinsics.json image.jpg --out result.json
```

## 📈 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GTX 1060 (6GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB free space
- **CPU**: 4 cores, 2.5GHz

### Recommended Requirements
- **GPU**: NVIDIA RTX 3080 (10GB VRAM) or better
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **CPU**: 8 cores, 3.0GHz

## 📝 Notes

- **Rust Code**: The Rust implementation in `localization_engine/` is for reference
- **Database**: Qdrant database persists between runs - no need to reprocess unless data changes
- **Scaling**: System can handle thousands of images and millions of 3D points
- **Accuracy**: Typical localization accuracy is appr. 15 centimeters with good feature matches

## 🤝 Support

For technical support or questions:
1. Check the troubleshooting section
2. Review log outputs with debug mode enabled
3. Verify all dependencies are correctly installed
4. Ensure GPU drivers and CUDA are properly configured

---

*This system provides state-of-the-art visual localization performance suitable for AR/VR applications, robotics, and autonomous navigation.*
