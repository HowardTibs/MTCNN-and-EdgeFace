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

def load_embeddings_from_database(database_dir="face_database"):
    """Load embeddings from the face database directory structure"""
    known_faces = {}
    
    # Try to load from combined directory first (most efficient)
    combined_dir = os.path.join(database_dir, "combined_embeddings")
    if os.path.exists(combined_dir):
        combined_catalog = os.path.join(combined_dir, "catalog.json")
        
        if os.path.exists(combined_catalog):
            # Load from combined catalog
            with open(combined_catalog, 'r') as f:
                catalog = json.load(f)
            
            for person_name, embedding_file in catalog.items():
                embedding_path = os.path.join(combined_dir, embedding_file)
                try:
                    known_faces[person_name] = np.load(embedding_path)
                    print(f"Loaded embedding for {person_name} from combined directory")
                except Exception as e:
                    print(f"Error loading {embedding_path}: {e}")
            
            if known_faces:
                return known_faces
    
    # If combined directory doesn't exist or is empty, try the master catalog
    master_catalog_file = os.path.join(database_dir, "embeddings_catalog.json")
    
    if os.path.exists(master_catalog_file):
        with open(master_catalog_file, 'r') as f:
            master_catalog = json.load(f)
        
        for person_name, info in master_catalog.items():
            embedding_path = os.path.join(info["embeddings_dir"], info["average_embedding"])
            try:
                known_faces[person_name] = np.load(embedding_path)
                print(f"Loaded embedding for {person_name} from master catalog")
            except Exception as e:
                print(f"Error loading {embedding_path}: {e}")
        
        if known_faces:
            return known_faces
    
    # If all else fails, search each person directory individually
    person_dirs = [d for d in os.listdir(database_dir) 
                   if os.path.isdir(os.path.join(database_dir, d)) 
                   and d != "combined_embeddings"]
    
    for person_name in person_dirs:
        person_dir = os.path.join(database_dir, person_name)
        embeddings_dir = os.path.join(person_dir, "embeddings")
        
        if not os.path.exists(embeddings_dir):
            continue
        
        # Try to find the average embedding first
        avg_embedding_path = os.path.join(embeddings_dir, f"{person_name}_average.npy")
        
        if os.path.exists(avg_embedding_path):
            try:
                known_faces[person_name] = np.load(avg_embedding_path)
                print(f"Loaded average embedding for {person_name}")
                continue
            except Exception as e:
                print(f"Error loading {avg_embedding_path}: {e}")
        
        # If no average embedding, try individual video embeddings
        person_catalog = os.path.join(embeddings_dir, "catalog.json")
        
        if os.path.exists(person_catalog):
            with open(person_catalog, 'r') as f:
                catalog = json.load(f)
            
            if "videos" in catalog:
                # Just use the first video embedding
                for video_name, video_info in catalog["videos"].items():
                    embedding_path = os.path.join(embeddings_dir, video_info["embedding_file"])
                    try:
                        known_faces[person_name] = np.load(embedding_path)
                        print(f"Loaded embedding for {person_name} from video {video_name}")
                        break
                    except Exception as e:
                        print(f"Error loading {embedding_path}: {e}")
        
        # If no catalog, just try to load any .npy file
        if person_name not in known_faces:
            npy_files = [f for f in os.listdir(embeddings_dir) if f.endswith(".npy")]
            
            if npy_files:
                try:
                    embedding_path = os.path.join(embeddings_dir, npy_files[0])
                    known_faces[person_name] = np.load(embedding_path)
                    print(f"Loaded embedding for {person_name} from {npy_files[0]}")
                except Exception as e:
                    print(f"Error loading {embedding_path}: {e}")
    
    return known_faces

def run_face_recognition(database_dir="face_database", 
                        model_path="edgeface_xs_gamma_06.pt", 
                        model_name="edgeface_xs_gamma_06",
                        recognition_threshold=0.7,
                        camera_id=0,
                        display_size=(640, 480)):
    """Run real-time face recognition using the database structure"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load known faces
    known_faces = load_embeddings_from_database(database_dir)
    if not known_faces:
        print("No embeddings found. Running in 'detection only' mode.")
    else:
        print(f"Loaded {len(known_faces)} face embeddings.")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera id: {camera_id}")
    
    # Set display size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
    
    print("Starting face recognition. Press 'q' to quit.")
    
    # For FPS calculation
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = cv2.getTickCount()
        else:
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(frame_rgb)
        
        # Draw bounding boxes and do recognition
        for face in faces:
            x, y, w, h = face['box']
            keypoints = face['keypoints']
            confidence = face['confidence']
            
            # Safety check for bounding box
            if w <= 0 or h <= 0:
                continue
            
            # Expand the face bounding box slightly
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
            
            # Compare with known faces using cosine similarity
            if known_faces:
                best_match = None
                best_similarity = -1
                
                for name, known_emb in known_faces.items():
                    # Calculate cosine similarity (dot product of normalized vectors)
                    similarity = np.dot(embedding, known_emb)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                
                # Decide if it's a recognized face or unknown
                if best_similarity > recognition_threshold:
                    label = f"{best_match} ({best_similarity:.2f})"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = f"Unknown ({best_similarity:.2f})"
                    color = (0, 0, 255)  # Red for unknown
            else:
                label = f"Face ({confidence:.2f})"
                color = (255, 255, 0)  # Yellow for detection-only mode
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw facial keypoints
            for kp in keypoints.values():
                cv2.circle(frame, kp, 2, (0, 255, 255), 2)
        
        # Show info text
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_text = f"Known faces: {len(known_faces)} | Threshold: {recognition_threshold:.2f}"
        cv2.putText(frame, info_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "q: Quit | +/-: Adjust threshold", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('EdgeFace Recognition', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            recognition_threshold = min(1.0, recognition_threshold + 0.05)
            print(f"Recognition threshold increased to: {recognition_threshold:.2f}")
        elif key == ord('-'):
            recognition_threshold = max(0.0, recognition_threshold - 0.05)
            print(f"Recognition threshold decreased to: {recognition_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time face recognition")
    parser.add_argument("--database_dir", type=str, default="face_database",
                        help="Face database directory")
    parser.add_argument("--model_path", type=str, default="edgeface_xs_gamma_06.pt",
                        help="Path to the EdgeFace model weights")
    parser.add_argument("--model_name", type=str, default="edgeface_xs_gamma_06",
                        help="Name of the EdgeFace model")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Recognition threshold (higher = stricter)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="Display width")
    parser.add_argument("--height", type=int, default=480,
                        help="Display height")
    
    args = parser.parse_args()
    
    run_face_recognition(database_dir=args.database_dir, 
                         model_path=args.model_path,
                         model_name=args.model_name,
                         recognition_threshold=args.threshold,
                         camera_id=args.camera,
                         display_size=(args.width, args.height))
