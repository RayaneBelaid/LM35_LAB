
import cv2
import numpy as np
import time  # Pour gérer les délais entre les détections

# Initialisation de la webcam
vidcap = cv2.VideoCapture(1)

# Vérifier si la webcam est ouverte
if not vidcap.isOpened():
    print("Erreur: La webcam n'a pas pu être ouverte.")
    exit()

# Initialisation du premier cadre pour la détection de mouvement
ret, prev_frame = vidcap.read()
if not ret:
    print("Erreur: Impossible de lire la première image de la caméra.")
    exit()

# Convertir l'image en niveau de gris
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

motion_count = 0
last_motion_time = 0  # Le temps du dernier mouvement détecté
motion_delay = 1  # Délai minimum entre deux détections de mouvement (en secondes)

while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    # Convertir l'image actuelle en niveau de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculer la différence absolue entre l'image actuelle et l'image précédente
    diff = cv2.absdiff(prev_gray, gray)

    # Augmenter le seuil de détection pour ne détecter que les changements plus importants
    _, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)

    # Appliquer un flou pour réduire le bruit
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Trouver les contours des zones en mouvement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si des contours sont détectés, cela signifie qu'il y a eu un mouvment
    detected_motion = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrer les petits mouvements
            detected_motion = True
            break  # Un seul contour suffira pour détecter un mouvement

    # Si un mouvement est détecté et que le délai est respecté
    current_time = time.time()  # Récupérer le temps actuel
    if detected_motion and (current_time - last_motion_time > motion_delay):
        motion_count += 1
        last_motion_time = current_time  # Mettre à jour le temps du dernier mouvement

    # Afficher le texte avec le nombre de mouvements détectés
    cv2.putText(frame, f"Nombre de mouvements: {motion_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher l'image actuelle
    cv2.imshow("Webcam", frame)

    # Mettre à jour l'image précédente pour la prochaine itération
    prev_gray = gray

    # Quitter la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres ouvertes
vidcap.release()
cv2.destroyAllWindows()






