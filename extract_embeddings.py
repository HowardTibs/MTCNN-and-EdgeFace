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
                                 max_frames=100, sampling_rate=1):
    """
    Extract face embeddings from a video file.

    Args:
        video_path: Path to the video file
        person_name: Name of the person in the video
        output_dir: Directory to save embeddings
        model: Loaded EdgeFace model
        detector: MTCNN detector
        transform: Image transform
        device: Torch device
        max_frames: Maximum number of frames to process
        sampling_rate: Process every Nth frame (default set to 1 to process all frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    embeddings = []    
    frame_count = 0
    processed_frames = 0
    
    pbar = tqdm(total=min(max_frames, total_frames // sampling_rate), 
                desc=f"Processing {os.path.basename(video_path)}")
    
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Process every Nth frame
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
            face = faces[0]
            x, y, w, h = face['box']
            if w <= 0 or h <= 0:
                continue
            
            # Expand the bounding box slightly to include more context
            margin = int(max(w, h) * 0.1)
            x_expanded = max(0, x - margin)
            y_expanded = max(0, y - margin)
            w_expanded = min(frame.shape[1] - x_expanded, w + 2 * margin)
            h_expanded = min(frame.shape[0] - y_expanded, h + 2 * margin)
            
            face_crop = frame_rgb[y_expanded:y_expanded+h_expanded, 
                                  x_expanded:x_expanded+w_expanded]
            if face_crop.size == 0:
                continue
            
            embedding = get_face_embedding(model, face_crop, transform, device)
            embeddings.append(embedding)
    
    pbar.close()
    cap.release()
    
    if not embeddings:
        print(f"No faces found in video: {video_path}")
        return None
    
    # Use median aggregation for robustness
    embeddings = np.array(embeddings)
    avg_embedding = np.median(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    embedding_file = os.path.join(output_dir, f"{person_name}.npy")
    np.save(embedding_file, avg_embedding)
    print(f"Saved embedding for {person_name} to {embedding_file}")
    
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
                          sampling_rate=1,
                          video_extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Process the entire face database directory structure.

    Args:
        database_dir: Root directory containing person subfolders with videos.
        model_path: Path to the EdgeFace model weights.
        model_name: Name of the EdgeFace model.
        max_frames: Maximum number of frames to process per video.
        sampling_rate: Process every Nth frame (default is 1 to process all frames).
        video_extensions: Tuple of valid video file extensions.
    """
    if not os.path.exists(database_dir):
        print(f"Database directory '{database_dir}' does not exist.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    detector = MTCNN()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    master_catalog = {}
    master_catalog_file = os.path.join(database_dir, "embeddings_catalog.json")
    if os.path.exists(master_catalog_file):
        with open(master_catalog_file, 'r') as f:
            master_catalog = json.load(f)
    
    person_dirs = [d for d in os.listdir(database_dir) 
                   if os.path.isdir(os.path.join(database_dir, d))]
    
    if not person_dirs:
        print(f"No person directories found in '{database_dir}'.")
        return
    
    print(f"Found {len(person_dirs)} person directories.")
    
    for person_name in person_dirs:
        person_dir = os.path.join(database_dir, person_name)
        embeddings_dir = os.path.join(person_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(person_dir) 
                       if os.path.isfile(os.path.join(person_dir, f)) 
                       and f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"No video files found for person '{person_name}'.")
            continue
        
        print(f"Processing {len(video_files)} videos for person '{person_name}'.")
        
        person_embeddings = []
        video_metadata = {}
        
        for video_file in video_files:
            video_path = os.path.join(person_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            print(f"Processing video: {video_file}")
            
            result = extract_embeddings_from_video(
                video_path, video_name, embeddings_dir,
                model, detector, transform, device,
                max_frames, sampling_rate
            )
            
            if result is not None:
                embedding, metadata = result
                person_embeddings.append(embedding)
                video_metadata[video_name] = metadata
        
        if person_embeddings:
            # Use median aggregation across videos for robustness
            person_embeddings = np.array(person_embeddings)
            person_avg_embedding = np.median(person_embeddings, axis=0)
            person_avg_embedding = person_avg_embedding / np.linalg.norm(person_avg_embedding)
            
            person_embedding_file = os.path.join(embeddings_dir, f"{person_name}_average.npy")
            np.save(person_embedding_file, person_avg_embedding)
            print(f"Saved average embedding for {person_name} to {person_embedding_file}")
            
            person_catalog_file = os.path.join(embeddings_dir, "catalog.json")
            person_catalog = {
                "person_name": person_name,
                "average_embedding": f"{person_name}_average.npy",
                "videos": video_metadata
            }
            
            with open(person_catalog_file, 'w') as f:
                json.dump(person_catalog, f, indent=2)
            
            master_catalog[person_name] = {
                "embeddings_dir": os.path.join(person_dir, "embeddings"),
                "average_embedding": f"{person_name}_average.npy",
                "video_count": len(video_metadata)
            }
    
    with open(master_catalog_file, 'w') as f:
        json.dump(master_catalog, f, indent=2)
    
    print(f"Processed {len(master_catalog)} people. Master catalog saved to {master_catalog_file}")
    
    combined_dir = os.path.join(database_dir, "combined_embeddings")
    os.makedirs(combined_dir, exist_ok=True)
    
    for person_name, info in master_catalog.items():
        source_path = os.path.join(info["embeddings_dir"], info["average_embedding"])
        dest_path = os.path.join(combined_dir, f"{person_name}.npy")
        if os.path.exists(source_path):
            embedding = np.load(source_path)
            np.save(dest_path, embedding)
            print(f"Copied {person_name}'s embedding to combined directory")
    
    combined_catalog = {person_name: f"{person_name}.npy" for person_name in master_catalog.keys()}
    combined_catalog_file = os.path.join(combined_dir, "catalog.json")
    with open(combined_catalog_file, 'w') as f:
        json.dump(combined_catalog, f, indent=2)
    
    print(f"Created combined embeddings directory with {len(combined_catalog)} embeddings")
    print("Face database processing complete!")

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
    parser.add_argument("--sampling_rate", type=int, default=1, 
                        help="Process every Nth frame (set to 1 to process all frames)")
    
    args = parser.parse_args()
    process_face_database(
        args.database_dir, args.model_path, args.model_name, 
        args.max_frames, args.sampling_rate
    )
