import cv2
import torch
import numpy as np
import os
import json
import argparse
from mtcnn.mtcnn import MTCNN
import torchvision.transforms as transforms
import timm
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# EdgeFace model definition
class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))

def replace_linear_with_lowrank(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            rank = max(2, int(min(module.in_features, module.out_features) * rank_ratio))
            bias = module.bias is not None
            setattr(model, name, LoRaLin(module.in_features, module.out_features, rank, bias))
        else:
            replace_linear_with_lowrank(module, rank_ratio)
    return model

class TimmFRWrapperV2(nn.Module):
    def __init__(self, model_name='edgenext_x_small', featdim=512):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.reset_classifier(featdim)

    def forward(self, x):
        return self.model(x)

def get_model(name):
    if name == 'edgeface_xs_gamma_06':
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_x_small'), rank_ratio=0.6)
    elif name == 'edgeface_s_gamma_05':
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_small'), rank_ratio=0.5)
    else:
        raise ValueError(f"Unknown model name: {name}")

# Helper function to get a face embedding
@torch.no_grad()
def get_face_embedding(model, face_crop, transform, device):
    face_pil = Image.fromarray(face_crop)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    emb = model(face_tensor)
    emb = emb.squeeze(0).cpu().numpy()
    # Normalize the embedding
    emb = emb / np.linalg.norm(emb)
    return emb

def extract_embeddings_from_video(video_path, person_name, output_dir, 
                                 model, detector, transform, device,
                                 max_frames=100, sampling_rate=30):
    """
    Extract face embeddings from a video file
    
    Args:
        video_path: Path to the video file
        person_name: Name of the person in the video
        output_dir: Directory to save embeddings
        model: Loaded EdgeFace model
        detector: MTCNN detector
        transform: Image transform
        device: Torch device
        max_frames: Maximum number of frames to process
        sampling_rate: Process every Nth frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Storage for all face embeddings
    embeddings = []
    
    # Process frames
    frame_count = 0
    processed_frames = 0
    
    # Create a progress bar
    pbar = tqdm(total=min(max_frames, total_frames // sampling_rate), 
                desc=f"Processing {os.path.basename(video_path)}")
    
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Skip frames according to sampling rate
        if frame_count % sampling_rate != 0:
            continue
        
        processed_frames += 1
        pbar.update(1)
        
        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(frame_rgb)
        
        if faces:
            # Sort faces by size (largest first), assuming the largest face is the main subject
            faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
            face = faces[0]  # Take the largest face
            
            x, y, w, h = face['box']
            if w <= 0 or h <= 0:
                continue
            
            # Expand the face bounding box slightly to include more context
            margin = int(max(w, h) * 0.1)
            x_expanded = max(0, x - margin)
            y_expanded = max(0, y - margin)
            w_expanded = min(frame.shape[1] - x_expanded, w + 2 * margin)
            h_expanded = min(frame.shape[0] - y_expanded, h + 2 * margin)
            
            # Crop the face
            face_crop = frame_rgb[y_expanded:y_expanded+h_expanded, 
                                  x_expanded:x_expanded+w_expanded]
            
            if face_crop.size == 0:
                continue
            
            # Get embedding
            embedding = get_face_embedding(model, face_crop, transform, device)
            embeddings.append(embedding)
    
    pbar.close()
    cap.release()
    
    if not embeddings:
        print(f"No faces found in video: {video_path}")
        return None
    
    # Calculate average embedding
    embeddings = np.array(embeddings)
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize again
    
    # Save the embedding
    embedding_file = os.path.join(output_dir, f"{person_name}.npy")
    np.save(embedding_file, avg_embedding)
    print(f"Saved embedding for {person_name} to {embedding_file}")
    
    # Return metadata
    metadata = {
        "embedding_file": f"{person_name}.npy",
        "source_video": video_path,
        "frames_processed": processed_frames,
        "total_faces_found": len(embeddings)
    }
    
    return avg_embedding, metadata

def process_face_database(database_dir="face_database", 
                          model_path="edgeface_xs_gamma_06.pt", 
                          model_name="edgeface_xs_gamma_06",
                          max_frames=100, 
                          sampling_rate=30,
                          video_extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Process the entire face database directory structure
    
    Args:
        database_dir: Root directory containing person subfolders with videos
        model_path: Path to the EdgeFace model weights
        model_name: Name of the EdgeFace model
        max_frames: Maximum number of frames to process per video
        sampling_rate: Process every Nth frame
        video_extensions: Tuple of valid video file extensions
    """
    # Check if database directory exists
    if not os.path.exists(database_dir):
        print(f"Database directory '{database_dir}' does not exist.")
        return
    
    # Load model (do this once to avoid reloading for each video)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize MTCNN detector (do this once)
    detector = MTCNN()
    
    # Define transform (do this once)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create a master catalog to store all embeddings info
    master_catalog = {}
    master_catalog_file = os.path.join(database_dir, "embeddings_catalog.json")
    
    # If master catalog exists, load it
    if os.path.exists(master_catalog_file):
        with open(master_catalog_file, 'r') as f:
            master_catalog = json.load(f)
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(database_dir) 
                   if os.path.isdir(os.path.join(database_dir, d))]
    
    if not person_dirs:
        print(f"No person directories found in '{database_dir}'.")
        return
    
    print(f"Found {len(person_dirs)} person directories.")
    
    # Process each person's directory
    for person_name in person_dirs:
        person_dir = os.path.join(database_dir, person_name)
        
        # Create embeddings directory inside the person's directory if it doesn't exist
        embeddings_dir = os.path.join(person_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Get all video files in the person's directory
        video_files = [f for f in os.listdir(person_dir) 
                      if os.path.isfile(os.path.join(person_dir, f)) 
                      and f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"No video files found for person '{person_name}'.")
            continue
        
        print(f"Processing {len(video_files)} videos for person '{person_name}'.")
        
        # Process each video
        person_embeddings = []
        video_metadata = {}
        
        for video_file in video_files:
            video_path = os.path.join(person_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            print(f"Processing video: {video_file}")
            
            # Extract embeddings from the video
            result = extract_embeddings_from_video(
                video_path, video_name, embeddings_dir,
                model, detector, transform, device,
                max_frames, sampling_rate
            )
            
            if result is not None:
                embedding, metadata = result
                person_embeddings.append(embedding)
                video_metadata[video_name] = metadata
        
        # If we found faces in at least one video, create an average embedding for the person
        if person_embeddings:
            # Calculate average embedding across all videos
            person_embeddings = np.array(person_embeddings)
            person_avg_embedding = np.mean(person_embeddings, axis=0)
            person_avg_embedding = person_avg_embedding / np.linalg.norm(person_avg_embedding)
            
            # Save the person's average embedding
            person_embedding_file = os.path.join(embeddings_dir, f"{person_name}_average.npy")
            np.save(person_embedding_file, person_avg_embedding)
            print(f"Saved average embedding for {person_name} to {person_embedding_file}")
            
            # Update person's catalog
            person_catalog_file = os.path.join(embeddings_dir, "catalog.json")
            person_catalog = {
                "person_name": person_name,
                "average_embedding": f"{person_name}_average.npy",
                "videos": video_metadata
            }
            
            with open(person_catalog_file, 'w') as f:
                json.dump(person_catalog, f, indent=2)
            
            # Update master catalog
            master_catalog[person_name] = {
                "embeddings_dir": os.path.join(person_dir, "embeddings"),
                "average_embedding": f"{person_name}_average.npy",
                "video_count": len(video_metadata)
            }
    
    # Save master catalog
    with open(master_catalog_file, 'w') as f:
        json.dump(master_catalog, f, indent=2)
    
    print(f"Processed {len(master_catalog)} people. Master catalog saved to {master_catalog_file}")
    
    # Create a combined embeddings directory at the root with links to all person embeddings
    combined_dir = os.path.join(database_dir, "combined_embeddings")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Copy all average embeddings to the combined directory
    for person_name, info in master_catalog.items():
        source_path = os.path.join(info["embeddings_dir"], info["average_embedding"])
        dest_path = os.path.join(combined_dir, f"{person_name}.npy")
        
        if os.path.exists(source_path):
            # Read the embedding and write it to the combined directory
            embedding = np.load(source_path)
            np.save(dest_path, embedding)
            print(f"Copied {person_name}'s embedding to combined directory")
    
    # Create a combined catalog
    combined_catalog = {person_name: f"{person_name}.npy" for person_name in master_catalog.keys()}
    combined_catalog_file = os.path.join(combined_dir, "catalog.json")
    
    with open(combined_catalog_file, 'w') as f:
        json.dump(combined_catalog, f, indent=2)
    
    print(f"Created combined embeddings directory with {len(combined_catalog)} embeddings")
    print(f"Face database processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process face database directory")
    parser.add_argument("--database_dir", type=str, default="face_database",
                        help="Root directory containing person subfolders with videos")
    parser.add_argument("--model_path", type=str, default="edgeface_xs_gamma_06.pt", 
                        help="Path to the EdgeFace model weights")
    parser.add_argument("--model_name", type=str, default="edgeface_xs_gamma_06", 
                        help="Name of the EdgeFace model")
    parser.add_argument("--max_frames", type=int, default=100, 
                        help="Maximum number of frames to process per video")
    parser.add_argument("--sampling_rate", type=int, default=30, 
                        help="Process every Nth frame")
    
    args = parser.parse_args()
    process_face_database(
        args.database_dir, args.model_path, args.model_name, 
        args.max_frames, args.sampling_rate
    )