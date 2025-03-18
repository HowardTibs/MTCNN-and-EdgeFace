import cv2
import torch
import numpy as np
import os
import json
import argparse
import time
from mtcnn.mtcnn import MTCNN
import torchvision.transforms as transforms
import timm
import torch.nn as nn
from PIL import Image
from collections import deque

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
    emb = emb / np.linalg.norm(emb)
    return emb

def load_embeddings_from_database(database_dir="face_database"):
    known_faces = {}
    combined_dir = os.path.join(database_dir, "combined_embeddings")
    if os.path.exists(combined_dir):
        combined_catalog = os.path.join(combined_dir, "catalog.json")
        if os.path.exists(combined_catalog):
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
    person_dirs = [d for d in os.listdir(database_dir) 
                   if os.path.isdir(os.path.join(database_dir, d)) and d != "combined_embeddings"]
    for person_name in person_dirs:
        person_dir = os.path.join(database_dir, person_name)
        embeddings_dir = os.path.join(person_dir, "embeddings")
        if not os.path.exists(embeddings_dir):
            continue
        avg_embedding_path = os.path.join(embeddings_dir, f"{person_name}_average.npy")
        if os.path.exists(avg_embedding_path):
            try:
                known_faces[person_name] = np.load(avg_embedding_path)
                print(f"Loaded average embedding for {person_name}")
                continue
            except Exception as e:
                print(f"Error loading {avg_embedding_path}: {e}")
        person_catalog = os.path.join(embeddings_dir, "catalog.json")
        if os.path.exists(person_catalog):
            with open(person_catalog, 'r') as f:
                catalog = json.load(f)
            if "videos" in catalog:
                for video_name, video_info in catalog["videos"].items():
                    embedding_path = os.path.join(embeddings_dir, video_info["embedding_file"])
                    try:
                        known_faces[person_name] = np.load(embedding_path)
                        print(f"Loaded embedding for {person_name} from video {video_name}")
                        break
                    except Exception as e:
                        print(f"Error loading {embedding_path}: {e}")
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

# Face detector with memory
class FaceDetectorWithMemory:
    def __init__(self, detector_confidence=0.7, skip_frames=1):
        self.detector = MTCNN()
        self.confidence_threshold = detector_confidence
        self.last_faces = None
        self.frame_count = 0
        self.skip_frames = skip_frames

    def detect_faces(self, frame_rgb):
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0 or self.last_faces is None:
            faces = self.detector.detect_faces(frame_rgb)
            faces = [face for face in faces if face['confidence'] >= self.confidence_threshold]
            self.last_faces = faces
            return faces
        else:
            return self.last_faces

# Identity smoothing with temporal voting
class IdentityTracker:
    def __init__(self, buffer_size=15, min_confidence=0.95, min_votes_ratio=0.5):
        self.face_history = {}  # track by face id
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.min_votes_ratio = min_votes_ratio

    def update_identity(self, face_id, name, similarity):
        if similarity < self.min_confidence:
            name = "Unknown"
        if face_id not in self.face_history:
            self.face_history[face_id] = deque(maxlen=self.buffer_size)
        self.face_history[face_id].append((name, similarity))
        return self.get_identity(face_id)

    def get_identity(self, face_id):
        if face_id not in self.face_history or not self.face_history[face_id]:
            return "Unknown", 0.0
        votes = {}
        avg_conf = {}
        for name, conf in self.face_history[face_id]:
            if name not in votes:
                votes[name] = 0
                avg_conf[name] = []
            votes[name] += 1
            avg_conf[name].append(conf)
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_name, top_votes = sorted_votes[0]
        if top_votes < len(self.face_history[face_id]) * self.min_votes_ratio:
            return "Unknown", 0.0
        conf = sum(avg_conf[top_name]) / len(avg_conf[top_name])
        return top_name, conf

def run_face_recognition(database_dir="face_database", 
                         model_path="edgeface_xs_gamma_06.pt", 
                         model_name="edgeface_xs_gamma_06",
                         recognition_threshold=0.95,
                         camera_id=0,
                         display_size=(640, 480),
                         detector_confidence=0.7,
                         process_every_n_frames=1):
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = get_model(model_name)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Initializing face detector...")
    detector = FaceDetectorWithMemory(detector_confidence=detector_confidence, skip_frames=process_every_n_frames)
    print("Initializing identity tracker...")
    tracker = IdentityTracker(buffer_size=10, min_confidence=recognition_threshold, min_votes_ratio=0.5)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading face database from {database_dir}...")
    known_faces = load_embeddings_from_database(database_dir)
    if not known_faces:
        print("No embeddings found. Running in 'detection only' mode.")
    else:
        print(f"Loaded {len(known_faces)} face embeddings.")

    next_face_id = 0
    active_face_tracks = {}

    print(f"Starting video capture on camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera id: {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

    frame_count = 0
    start_time = time.time()
    fps = 0

    # New strict matching parameters
    strict_threshold = 0.95
    margin_threshold = 0.15

    # Simulation parameter for luminance
    simulated_brightness = 1.0  # 1.0 means no brightness change

    print("Starting face recognition. Press 'q' to quit.")
    print("Simulation keys: 'b' (increase brightness), 'n' (decrease brightness)")
    print("Other keys: '+/-': Adjust threshold | 'd/D': Adjust detector confidence | 'f/F': Adjust frame skipping")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error reading frame. Check camera connection.")
            break

        # Adjust brightness based on simulated_brightness
        frame = cv2.convertScaleAbs(frame, alpha=simulated_brightness, beta=0)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)
        current_centers = {}

        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            if confidence < detector_confidence:
                continue
            if w <= 0 or h <= 0:
                continue

            center_x, center_y = x + w // 2, y + h // 2

            # Track the face by matching centers
            matched_id = None
            for face_id, (old_x, old_y) in active_face_tracks.items():
                distance = np.sqrt((center_x - old_x) ** 2 + (center_y - old_y) ** 2)
                if distance < max(w, h) * 0.5:
                    matched_id = face_id
                    break
            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1

            active_face_tracks[matched_id] = (center_x, center_y)
            current_centers[matched_id] = (center_x, center_y)

            # Perform recognition if embeddings exist
            if known_faces:
                margin = int(max(w, h) * 0.2)
                x_expanded = max(0, x - margin)
                y_expanded = max(0, y - margin)
                w_expanded = min(frame.shape[1] - x_expanded, w + 2 * margin)
                h_expanded = min(frame.shape[0] - y_expanded, h + 2 * margin)
                face_crop = frame_rgb[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]
                if face_crop.size == 0:
                    continue

                # Get the face embedding
                embedding = get_face_embedding(model, face_crop, transform, device)

                # Compute cosine similarities with all known faces
                similarity_list = []
                for known_name, known_emb in known_faces.items():
                    sim = np.dot(embedding, known_emb)
                    similarity_list.append((known_name, sim))
                similarity_list.sort(key=lambda x: x[1], reverse=True)

                # Strict matching: require high similarity and a clear margin
                if len(similarity_list) > 1:
                    best_name, best_similarity = similarity_list[0]
                    second_similarity = similarity_list[1][1]
                    if best_similarity < strict_threshold or (best_similarity - second_similarity) < margin_threshold:
                        best_name = "Unknown"
                        best_similarity = 0.0
                else:
                    best_name, best_similarity = similarity_list[0]
                    if best_similarity < strict_threshold:
                        best_name = "Unknown"
                        best_similarity = 0.0

                name, confidence_score = tracker.update_identity(matched_id, best_name, best_similarity)
            else:
                name, confidence_score = tracker.get_identity(matched_id) if matched_id in active_face_tracks else ("Unknown", 0)

            if name != "Unknown":
                label = f"{name} ({confidence_score:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Unknown ({confidence_score:.2f})" if confidence_score > 0 else "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if 'keypoints' in face:
                for kp in face['keypoints'].values():
                    cv2.circle(frame, kp, 2, (0, 255, 255), 1)

        for face_id in list(active_face_tracks.keys()):
            if face_id not in current_centers:
                del active_face_tracks[face_id]

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        info_text = f"Known faces: {len(known_faces)} | Threshold: {recognition_threshold:.2f}"
        cv2.putText(frame, info_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        debug_text = f"Detector conf: {detector_confidence:.2f} | Faces: {len(faces)}"
        cv2.putText(frame, debug_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        instructions = ("q: Quit | +/-: Adjust threshold | d/D: Adjust detector confidence | "
                        "f/F: Adjust frame skipping | b/n: Adjust brightness")
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('EdgeFace Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            recognition_threshold = min(1.0, recognition_threshold + 0.05)
            tracker.min_confidence = recognition_threshold
            print(f"Recognition threshold increased to: {recognition_threshold:.2f}")
        elif key == ord('-'):
            recognition_threshold = max(0.0, recognition_threshold - 0.05)
            tracker.min_confidence = recognition_threshold
            print(f"Recognition threshold decreased to: {recognition_threshold:.2f}")
        elif key == ord('d'):
            detector_confidence = max(0.4, detector_confidence - 0.05)
            detector.confidence_threshold = detector_confidence
            print(f"Detector confidence decreased to: {detector_confidence:.2f}")
        elif key == ord('D'):
            detector_confidence = min(0.95, detector_confidence + 0.05)
            detector.confidence_threshold = detector_confidence
            print(f"Detector confidence increased to: {detector_confidence:.2f}")
        elif key == ord('f'):
            detector.skip_frames = min(10, detector.skip_frames + 1)
            print(f"Increased frame skipping: processing every {detector.skip_frames} frame(s)")
        elif key == ord('F'):
            detector.skip_frames = max(1, detector.skip_frames - 1)
            print(f"Decreased frame skipping: processing every {detector.skip_frames} frame(s)")
        elif key == ord('b'):
            simulated_brightness = min(3.0, simulated_brightness + 0.1)
            print(f"Simulated brightness increased to: {simulated_brightness:.2f}")
        elif key == ord('n'):
            simulated_brightness = max(0.1, simulated_brightness - 0.1)
            print(f"Simulated brightness decreased to: {simulated_brightness:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Run real-time face recognition")
    parser.add_argument("--database_dir", type=str, default="face_database",
                        help="Face database directory")
    parser.add_argument("--model_path", type=str, default="edgeface_xs_gamma_06.pt",
                        help="Path to the EdgeFace model weights")
    parser.add_argument("--model_name", type=str, default="edgeface_xs_gamma_06",
                        help="Name of the EdgeFace model")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Recognition threshold (higher = stricter)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--width", type=int, default=640,
                        help="Display width")
    parser.add_argument("--height", type=int, default=480,
                        help="Display height")
    parser.add_argument("--detector_confidence", type=float, default=0.7,
                        help="Face detector confidence threshold")
    parser.add_argument("--process_frames", type=int, default=1,
                        help="Process every N frames for recognition")
    
    args = parser.parse_args()
    
    run_face_recognition(database_dir=args.database_dir, 
                         model_path=args.model_path,
                         model_name=args.model_name,
                         recognition_threshold=args.threshold,
                         camera_id=args.camera,
                         display_size=(args.width, args.height),
                         detector_confidence=args.detector_confidence,
                         process_every_n_frames=args.process_frames)
