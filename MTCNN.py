import cv2
import torch
import numpy as np
import os
import json
from mtcnn.mtcnn import MTCNN
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import argparse
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
    face_tensor = transform(face_crop).unsqueeze(0).to(device)
    emb = model(face_tensor)
    emb = emb.squeeze(0).cpu().numpy()
    # Normalize the embedding
    emb = emb / np.linalg.norm(emb)
    return emb

def load_embeddings(embeddings_dir="embeddings"):
    """Load all embeddings from the specified directory"""
    known_faces = {}
    
    # Check if directory exists
    if not os.path.exists(embeddings_dir):
        print(f"Embeddings directory {embeddings_dir} does not exist.")
        return known_faces
    
    # Check if catalog exists
    catalog_file = os.path.join(embeddings_dir, "catalog.json")
    if not os.path.exists(catalog_file):
        print(f"Catalog file {catalog_file} does not exist.")
        # Try to load .npy files directly
        for file in os.listdir(embeddings_dir):
            if file.endswith(".npy"):
                name = os.path.splitext(file)[0]
                embedding_path = os.path.join(embeddings_dir, file)
                try:
                    known_faces[name] = np.load(embedding_path)
                    print(f"Loaded embedding for {name} from {embedding_path}")
                except Exception as e:
                    print(f"Error loading {embedding_path}: {e}")
        return known_faces
    
    # Load from catalog
    with open(catalog_file, 'r') as f:
        catalog = json.load(f)
    
    for name, info in catalog.items():
        embedding_path = os.path.join(embeddings_dir, info["embedding_file"])
        try:
            known_faces[name] = np.load(embedding_path)
            print(f"Loaded embedding for {name} from {embedding_path}")
        except Exception as e:
            print(f"Error loading {embedding_path}: {e}")
    
    return known_faces

def run_face_recognition(embeddings_dir="embeddings", 
                        model_path="edgeface_xs_gamma_06.pt", 
                        model_name="edgeface_xs_gamma_06",
                        recognition_threshold=0.6,
                        camera_id=0,
                        display_size=(640, 480)):
    """Run real-time face recognition"""
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
    known_faces = load_embeddings(embeddings_dir)
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(frame_rgb)
        
        # Draw bounding boxes and do recognition
        for face in faces:
            x, y, w, h = face['box']
            keypoints = face['keypoints']
            
            # Safety check for bounding box
            if w <= 0 or h <= 0:
                continue
            
            # Crop the face
            face_crop = frame_rgb[y:y+h, x:x+w]
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
                label = "Face Detected"
                color = (255, 255, 0)  # Yellow for detection-only mode
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw facial keypoints
            for kp in keypoints.values():
                cv2.circle(frame, kp, 2, (0, 255, 255), 2)
        
        # Show number of known faces and recognition threshold
        info_text = f"Known faces: {len(known_faces)} | Threshold: {recognition_threshold:.2f}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Real-Time Face Recognition', frame)
        
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
    parser.add_argument("--embeddings_dir", type=str, default="embeddings",
                        help="Directory containing face embeddings")
    parser.add_argument("--model_path", type=str, default="edgeface_xs_gamma_06.pt",
                        help="Path to the EdgeFace model weights")
    parser.add_argument("--model_name", type=str, default="edgeface_xs_gamma_06",
                        help="Name of the EdgeFace model")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Recognition threshold (higher = stricter)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="Display width")
    parser.add_argument("--height", type=int, default=480,
                        help="Display height")
    
    args = parser.parse_args()
    run_face_recognition(
        args.embeddings_dir, args.model_path, args.model_name,
        args.threshold, args.camera, (args.width, args.height)
    )