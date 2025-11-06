import cv2
import numpy as np
from google.colab.patches import cv2_imshow  

# Carrega a imagem
img = cv2.imread('imgCarro2.png')

# Converte para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Para manter a lógica original, vamos aplicar threshold em um único canal do HSV
# Aqui escolhemos o canal H (Hue), que representa a cor
hue_channel = hsv[:, :, 0]

# Aplica threshold no canal H
ret, thresh = cv2.threshold(hue_channel, 60, 255, cv2.THRESH_BINARY)  
# 60 é um valor arbitrário para separar tons (verde ~ 35-85 no HSV)

# Encontra contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Seleciona o maior contorno
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Calcula o Convex Hull
    hull = cv2.convexHull(largest_contour)

    # Desenha o Convex Hull na imagem original
    cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)

# Exibe resultado
cv2_imshow(img)