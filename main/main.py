import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

# === Définition du modèle ===
class FacialCNN(nn.Module):
    def __init__(self, num_classes=8):  # 8 émotions typiques
        super(FacialCNN, self).__init__()

        # Bloc convolutionnel 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 canal d'entrée (grayscale)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Réduction: 48x48 -> 24x24
        )

        # Bloc convolutionnel 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Réduction: 24x24 -> 12x12
        )

        # Bloc convolutionnel 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Réduction: 12x12 -> 6x6
        )

        # Couches fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Réduction du surapprentissage
            nn.Linear(128 * 6 * 6, 512),  # 128 canaux * 6x6
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Passer à travers les couches convolutionnelles
        x = self.conv1(x)  # [batch, 32, 24, 24]
        x = self.conv2(x)  # [batch, 64, 12, 12]
        x = self.conv3(x)  # [batch, 128, 6, 6]

        # Aplatir pour les couches fully connected
        x = x.view(x.size(0), -1)  # [batch, 128*6*6]

        # Classification finale
        x = self.classifier(x)  # [batch, num_classes]

        return x

# === Chargement du modèle entraîné ===
model = FacialCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# === Prétraitement des images ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# === Étiquettes d’émotions ===
labels = ['Happy', 'Disgust', 'Fear', 'Angry', 'Sad', 'Surprise', 'Neutral', 'Contempt']

# === Activation de la caméra ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        img = transform(face).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            emotion = labels[predicted.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Détection des émotions", frame)

    # Quitter avec la touche Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()