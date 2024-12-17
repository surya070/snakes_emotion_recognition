import sys
import cv2
import numpy as np
import pygame
import random
from keras.models import load_model

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load pre-trained emotion recognition model
model = load_model('C:/Users/mahes/Rvu-Python/Identify emotions/model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y + h, x:x + w]
        face = preprocess_face(face)
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion
    return None

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Initialize Pygame
pygame.init()

# Screen Dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Square Dodge")

# Player settings 
player_size = 50
player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - 2 * player_size]

# Enemy settings
enemy_size = 50
enemy_list = []

# Speed settings
SPEED = 10

# Clock 
clock = pygame.time.Clock()

# Font setting
font = pygame.font.SysFont("monospace", 35)

# Score
score = 0

# Current emotion
current_emotion = 'Neutral'

# Function to update the position of enemies
def drop_enemies(enemy_list):
    if len(enemy_list) < 3:
        x_pos = random.randint(0, SCREEN_WIDTH - enemy_size)
        y_pos = 0
        enemy_list.append([x_pos, y_pos, 'down'])

        x_pos = 0
        y_pos = random.randint(0, SCREEN_HEIGHT - enemy_size)
        enemy_list.append([x_pos, y_pos, 'right'])

        x_pos = SCREEN_WIDTH - enemy_size
        y_pos = random.randint(0, SCREEN_HEIGHT - enemy_size)
        enemy_list.append([x_pos, y_pos, 'left'])

def draw_enemies(enemy_list):
    for enemy_pos in enemy_list:
        pygame.draw.rect(screen, BLUE, (enemy_pos[0], enemy_pos[1], enemy_size, enemy_size))

def update_enemy_position(enemy_list, score):
    for idx, enemy_pos in enumerate(enemy_list):
        if enemy_pos[2] == 'down' and enemy_pos[1] >= 0 and enemy_pos[1] < SCREEN_HEIGHT:
            enemy_pos[1] += SPEED
        elif enemy_pos[2] == 'right' and enemy_pos[0] >= 0 and enemy_pos[0] < SCREEN_WIDTH:
            enemy_pos[0] += SPEED
        elif enemy_pos[2] == 'left' and enemy_pos[0] >= 0 and enemy_pos[0] < SCREEN_WIDTH:
            enemy_pos[0] -= SPEED
        else:
            enemy_list.pop(idx)
            score += 1
    return score

def collision_check(enemy_list, player_pos):
    for enemy_pos in enemy_list:
        if detect_collision(enemy_pos, player_pos):
            return True
    return False
    
def detect_collision(player_pos, enemy_pos):
    p_x = player_pos[0]
    p_y = player_pos[1]

    e_x = enemy_pos[0]
    e_y = enemy_pos[1]

    if (e_x >= p_x and e_x < (p_x + player_size)) or (p_x >= e_x and p_x < (e_x + enemy_size)):
        if (e_y >= p_y and e_y < (p_y + player_size)) or (p_y >= e_y and p_y < (e_y + enemy_size)):
            return True
    return False


game_over = False

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
    
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and player_pos[0] > 0:
        player_pos[0] -= 10
    if keys[pygame.K_RIGHT] and player_pos[0] < SCREEN_WIDTH - player_size:
        player_pos[0] += 10
    if keys[pygame.K_UP] and player_pos[1] > 0:
        player_pos[1] -= 10
    if keys[pygame.K_DOWN] and player_pos[1] < SCREEN_HEIGHT - player_size:
        player_pos[1] += 10

    ret, frame = cap.read()
    if not ret:
        break

    try:
        emotion = predict_emotion(frame)
        if emotion:
            current_emotion = emotion
            if current_emotion in ['Sad', 'Neutral']:
                SPEED = 15
            elif current_emotion in ['Surprise', 'Angry', 'Fear']:
                SPEED = 10
            elif current_emotion in ['Happy','Disgust']:
                SPEED = 20
    except Exception as e:
        print(f"Error: {e}")
    
    screen.fill(BLACK)

    drop_enemies(enemy_list)
    score = update_enemy_position(enemy_list, score)
    draw_enemies(enemy_list)

    text = font.render("Score: {0}".format(score), 1, WHITE)
    screen.blit(text, (SCREEN_WIDTH - 210, SCREEN_HEIGHT - 40))

    emotion_text = font.render("Emotion: {0}".format(current_emotion), 1, WHITE)
    screen.blit(emotion_text, (SCREEN_WIDTH - 350, 10))

    if collision_check(enemy_list, player_pos):
        game_over = True
        break

    pygame.draw.rect(screen, RED, (player_pos[0], player_pos[1], player_size, player_size))

    clock.tick(30)

    pygame.display.update()
    
cap.release()
cv2.destroyAllWindows()
pygame.quit()
