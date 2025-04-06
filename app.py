import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import math
import numpy as np
from threading import Thread
import queue
import requests  # For interacting with Ollama API

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect up to 2 hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
display = (800, 600)
screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -10)

# Variables for object and camera control
object_rotation = [0, 0, 0]  # Rotation angles for the object
object_scale = 1.0  # Scale of the object
camera_angle = 0  # Camera rotation angle
camera_zoom = -10  # Camera zoom level
light_position = [5, 5, 5, 1]  # Light source position

# Queue for hand tracking data
hand_data_queue = queue.Queue()

# List of shapes and current shape index
shapes = ["cube", "sphere", "pyramid", "torus", "cylinder", "cone", "house", "car"]
current_shape_index = 0

# Function to draw a glowing cube
def draw_cube():
    vertices = [
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1]
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]
    glBegin(GL_LINES)
    glColor3f(1.0, 0.5, 0.0)  # Bright orange color
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Function to draw a sphere
def draw_sphere():
    quadric = gluNewQuadric()
    gluSphere(quadric, 1, 32, 32)

# Function to draw a pyramid
def draw_pyramid():
    vertices = [
        [0, 1, 0],
        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, -1]
    ]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1)
    ]
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.5)  # Bright green color
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Function to draw a torus
def draw_torus():
    quadric = gluNewQuadric()
    gluDisk(quadric, 0.5, 1, 32, 32)

# Function to draw a cylinder
def draw_cylinder():
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.5, 0.5, 2, 32, 32)

# Function to draw a cone
def draw_cone():
    quadric = gluNewQuadric()
    gluCylinder(quadric, 0.5, 0, 2, 32, 32)

# Function to draw a house
def draw_house():
    # Base
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.6, 0.4)  # Brown color
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glEnd()

    # Roof
    glBegin(GL_TRIANGLES)
    glColor3f(0.5, 0.2, 0.1)  # Dark brown color
    glVertex3f(-1.5, -1, -1.5)
    glVertex3f(0, 1, 0)
    glVertex3f(1.5, -1, -1.5)
    glEnd()

# Function to draw a car
def draw_car():
    # Body
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.6, 0.8)  # Blue color
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(-1, 0, -1)
    glEnd()

    # Wheels
    glColor3f(0.1, 0.1, 0.1)  # Black color
    glPushMatrix()
    glTranslatef(-0.75, -1, -1)
    gluSphere(gluNewQuadric(), 0.25, 16, 16)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.75, -1, -1)
    gluSphere(gluNewQuadric(), 0.25, 16, 16)
    glPopMatrix()

# Function to detect hand gesture
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    if distance < 0.05:
        return "pinch"  # Pinching gesture
    elif distance > 0.2:
        return "open"  # Open hand gesture
    else:
        return "fist"  # Fist gesture

# Function to process hand tracking in a separate thread
def hand_tracking_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
    if not cap.isOpened():
        print("Error: Could not open laptop camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to lower resolution for faster processing
        frame = cv2.resize(frame, (320, 240))

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # Detect gesture and hand position
        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the center of the hand (palm base)
                palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                target_x = palm_base.x * 2 - 1  # Map to OpenGL coordinates (-1 to 1)
                target_y = -(palm_base.y * 2 - 1)  # Invert y-axis for OpenGL
                target_z = palm_base.z  # Use z for depth

                # Detect gesture
                gesture = detect_gesture(hand_landmarks)

                # Add hand data to the list
                hand_data.append((target_x, target_y, target_z, gesture))

            # Put hand data in the queue
            hand_data_queue.put(hand_data)

        # Exit on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to interact with Ollama
def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # Ollama API endpoint
            json={"model": "ollama3.2:7b", "prompt": prompt}
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

# Start hand tracking thread
Thread(target=hand_tracking_thread, daemon=True).start()

# Draw a glowing grid floor
def draw_floor():
    glBegin(GL_LINES)
    glColor3f(0.2, 0.2, 0.2)
    for i in range(-10, 11):
        glVertex3f(i, -1, -10)
        glVertex3f(i, -1, 10)
        glVertex3f(-10, -1, i)
        glVertex3f(10, -1, i)
    glEnd()

# Function to draw the HUD
def draw_hud():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, display[0], display[1], 0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Display current shape
    font = pygame.font.SysFont("Arial", 24)
    text = font.render(f"Shape: {shapes[current_shape_index]}", True, (255, 255, 255))
    text_data = pygame.image.tostring(text, "RGBA", True)
    glRasterPos2d(10, 20)
    glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Display active gestures
    text = font.render(f"Gesture: {gesture}", True, (255, 255, 255))
    text_data = pygame.image.tostring(text, "RGBA", True)
    glRasterPos2d(10, 50)
    glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Display Ollama's suggestion
    if ollama_suggestion:
        text = font.render(f"Ollama: {ollama_suggestion}", True, (255, 255, 255))
        text_data = pygame.image.tostring(text, "RGBA", True)
        glRasterPos2d(10, 80)
        glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# Main loop
current_shape_index = 0
target_x, target_y, target_z = 0, 0, 0  # Target position for gestures
gesture = "open"
ollama_suggestion = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Get hand data from the queue
    if not hand_data_queue.empty():
        hand_data = hand_data_queue.get()
        if len(hand_data) == 1:
            # One hand detected
            target_x, target_y, target_z, gesture = hand_data[0]
            if gesture == "pinch":
                object_scale = max(0.5, object_scale - 0.05)  # Make smaller
            elif gesture == "open":
                object_scale = min(2.0, object_scale + 0.05)  # Make bigger
            elif gesture == "fist":
                object_scale = 1.0  # Reset size
        elif len(hand_data) == 2:
            # Two hands detected
            _, _, _, gesture1 = hand_data[0]
            _, _, _, gesture2 = hand_data[1]
            if gesture1 == "open" and gesture2 == "open":
                # Rotate the object using the second hand
                object_rotation[1] += 2  # Rotate around the y-axis
            elif gesture1 == "pinch" and gesture2 == "pinch":
                # Cycle through shapes
                current_shape_index = (current_shape_index + 1) % len(shapes)

    # Ask Ollama for suggestions
    if gesture:
        ollama_suggestion = ask_ollama(f"Interpret this gesture for a 3D object manipulation program: {gesture}")

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    # Draw the floor
    draw_floor()

    # Set up object transformation
    glPushMatrix()
    glTranslatef(target_x, target_y, target_z)
    glRotatef(object_rotation[0], 1, 0, 0)
    glRotatef(object_rotation[1], 0, 1, 0)
    glRotatef(object_rotation[2], 0, 0, 1)
    glScalef(object_scale, object_scale, object_scale)

    # Draw the current shape
    if shapes[current_shape_index] == "cube":
        draw_cube()
    elif shapes[current_shape_index] == "sphere":
        draw_sphere()
    elif shapes[current_shape_index] == "pyramid":
        draw_pyramid()
    elif shapes[current_shape_index] == "torus":
        draw_torus()
    elif shapes[current_shape_index] == "cylinder":
        draw_cylinder()
    elif shapes[current_shape_index] == "cone":
        draw_cone()
    elif shapes[current_shape_index] == "house":
        draw_house()
    elif shapes[current_shape_index] == "car":
        draw_car()

    glPopMatrix()

    # Draw the HUD
    draw_hud()

    # Update the display
    pygame.display.flip()
    pygame.time.wait(10)

# Release resources
pygame.quit()