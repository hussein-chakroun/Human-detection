import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import math
import numpy as np
from threading import Thread, Event
import queue
import requests
import time
import sys
import os

class HandGestureApp:
    def __init__(self):
        # Configuration
        self.config = {
            'display': (800, 600),
            'max_hands': 2,
            'camera_index': 0,
            'camera_resolution': (320, 240),
            'ollama_endpoint': "http://localhost:11434/api/generate",
            'ollama_model': "ollama3.2:7b",
            'frame_rate': 60
        }
        
        # Application state
        self.object_rotation = [0, 0, 0]  # Rotation angles for the object
        self.object_scale = 1.0  # Scale of the object
        self.camera_angle = 0  # Camera rotation angle
        self.camera_zoom = -10  # Camera zoom level
        self.light_position = [5, 5, 5, 1]  # Light source position
        self.current_shape_index = 0
        self.target_x, self.target_y, self.target_z = 0, 0, 0
        self.gesture = "none"
        self.ollama_suggestion = None
        self.stop_event = Event()
        self.hand_data_queue = queue.Queue()
        
        # Available shapes
        self.shapes = ["cube", "sphere", "pyramid", "torus", "cylinder", "cone", "house", "car"]
        
        # Shape drawing functions
        self.shape_functions = {
            "cube": self.draw_cube,
            "sphere": self.draw_sphere,
            "pyramid": self.draw_pyramid,
            "torus": self.draw_torus,
            "cylinder": self.draw_cylinder,
            "cone": self.draw_cone,
            "house": self.draw_house,
            "car": self.draw_car
        }
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.config['max_hands'],
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def init_pygame(self):
        """Initialize Pygame and OpenGL"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.config['display'], DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Object Manipulation with Hand Gestures")
        self.clock = pygame.time.Clock()
        
        # Setup OpenGL perspective
        gluPerspective(45, (self.config['display'][0] / self.config['display'][1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -10)
        glEnable(GL_DEPTH_TEST)
        
        # Set up basic lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)
        
        # Enable color tracking
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def draw_cube(self):
        """Draw a wireframe cube with improved edges"""
        vertices = [
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw wireframe
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.5, 0.0)  # Bright orange color
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        
        # Reset line width
        glLineWidth(1.0)

    def draw_sphere(self):
        """Draw a sphere with improved quality"""
        glColor3f(0.0, 0.7, 0.9)  # Blue color
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluSphere(quadric, 1, 32, 32)
        gluDeleteQuadric(quadric)

    def draw_pyramid(self):
        """Draw a wireframe pyramid"""
        vertices = [
            [0, 1, 0], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]
        ]
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (2, 3), (3, 4), (4, 1)
        ]
        
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.5)  # Green color
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        glLineWidth(1.0)

    def draw_torus(self):
        """Draw a better torus"""
        glColor3f(1.0, 0.3, 0.6)  # Pink color
        
        # Draw a better torus using quadric objects
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        
        # Render as a series of disks to simulate a torus
        slices = 32
        for i in range(slices):
            angle = (i / slices) * 2 * np.pi
            glPushMatrix()
            glTranslatef(0.75 * np.cos(angle), 0, 0.75 * np.sin(angle))
            glRotatef(90, 0, 1, 0)
            gluDisk(quadric, 0, 0.25, 16, 1)
            glPopMatrix()
            
        gluDeleteQuadric(quadric)

    def draw_cylinder(self):
        """Draw a cylinder with improved appearance"""
        glColor3f(0.8, 0.8, 0.2)  # Yellow color
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluCylinder(quadric, 0.5, 0.5, 2, 32, 8)
        gluDeleteQuadric(quadric)

    def draw_cone(self):
        """Draw a cone with improved appearance"""
        glColor3f(1.0, 0.3, 0.0)  # Orange color
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluCylinder(quadric, 0.5, 0, 2, 32, 8)
        gluDeleteQuadric(quadric)

    def draw_house(self):
        """Draw a house with improved appearance"""
        # Base
        glLineWidth(2.0)
        glColor3f(0.8, 0.6, 0.4)  # Brown color
        
        # Draw the base as a wireframe cube
        glPushMatrix()
        glScalef(1, 0.8, 1)
        self.draw_cube()
        glPopMatrix()
        
        # Roof
        glBegin(GL_LINE_LOOP)
        glColor3f(0.5, 0.2, 0.1)  # Dark brown color
        glVertex3f(-1.2, 0.8, -1.2)
        glVertex3f(1.2, 0.8, -1.2)
        glVertex3f(0, 1.8, 0)
        glEnd()
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-1.2, 0.8, 1.2)
        glVertex3f(1.2, 0.8, 1.2)
        glVertex3f(0, 1.8, 0)
        glEnd()
        
        glBegin(GL_LINES)
        glVertex3f(-1.2, 0.8, -1.2)
        glVertex3f(-1.2, 0.8, 1.2)
        glVertex3f(1.2, 0.8, -1.2)
        glVertex3f(1.2, 0.8, 1.2)
        glEnd()
        
        glLineWidth(1.0)

    def draw_car(self):
        """Draw a car with improved appearance"""
        glLineWidth(2.0)
        glColor3f(0.2, 0.6, 0.8)  # Blue color
        
        # Body - draw as lines for wireframe look
        glBegin(GL_LINE_LOOP)
        glVertex3f(-1.5, -0.5, 1)
        glVertex3f(1.5, -0.5, 1)
        glVertex3f(1.5, 0.5, 1)
        glVertex3f(0, 0.5, 1)
        glVertex3f(-0.5, 1, 1)
        glVertex3f(-1.5, 1, 1)
        glEnd()
        
        # Other side
        glBegin(GL_LINE_LOOP)
        glVertex3f(-1.5, -0.5, -1)
        glVertex3f(1.5, -0.5, -1)
        glVertex3f(1.5, 0.5, -1)
        glVertex3f(0, 0.5, -1)
        glVertex3f(-0.5, 1, -1)
        glVertex3f(-1.5, 1, -1)
        glEnd()
        
        # Connect sides
        glBegin(GL_LINES)
        glVertex3f(-1.5, -0.5, 1)
        glVertex3f(-1.5, -0.5, -1)
        
        glVertex3f(1.5, -0.5, 1)
        glVertex3f(1.5, -0.5, -1)
        
        glVertex3f(1.5, 0.5, 1)
        glVertex3f(1.5, 0.5, -1)
        
        glVertex3f(0, 0.5, 1)
        glVertex3f(0, 0.5, -1)
        
        glVertex3f(-0.5, 1, 1)
        glVertex3f(-0.5, 1, -1)
        
        glVertex3f(-1.5, 1, 1)
        glVertex3f(-1.5, 1, -1)
        glEnd()

        # Wheels
        glColor3f(0.1, 0.1, 0.1)  # Black color
        self.draw_wheel(-1, -0.5, 1)
        self.draw_wheel(1, -0.5, 1)
        self.draw_wheel(-1, -0.5, -1)
        self.draw_wheel(1, -0.5, -1)
        
        glLineWidth(1.0)
    
    def draw_wheel(self, x, y, z):
        """Helper method to draw car wheels"""
        glPushMatrix()
        glTranslatef(x, y, z)
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluSphere(quadric, 0.25, 8, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def draw_floor(self):
        """Draw a grid floor with depth effect"""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        
        # Draw grid with distance-based coloring
        for i in range(-10, 11):
            # Intensity based on distance from center
            intensity = 0.1 + 0.1 * (1 - abs(i) / 10)
            
            # X-axis lines
            glColor3f(intensity, intensity, intensity * 1.5)
            glVertex3f(i, -1, -10)
            glVertex3f(i, -1, 10)
            
            # Z-axis lines
            glColor3f(intensity, intensity, intensity * 1.5)
            glVertex3f(-10, -1, i)
            glVertex3f(10, -1, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_hud(self):
        """Draw the heads-up display with information"""
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.config['display'][0], self.config['display'][1], 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Set up background for text
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0, 0, 0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(5, 5)
        glVertex2f(250, 5)
        glVertex2f(250, 100)
        glVertex2f(5, 100)
        glEnd()
        glDisable(GL_BLEND)

        # Display information with better font
        font = pygame.font.SysFont("Arial", 18, bold=True)
        
        # Current shape
        text = font.render(f"Shape: {self.shapes[self.current_shape_index]}", True, (255, 255, 255))
        text_data = pygame.image.tostring(text, "RGBA", True)
        glRasterPos2d(10, 20)
        glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Active gesture
        text = font.render(f"Gesture: {self.gesture}", True, (255, 255, 255))
        text_data = pygame.image.tostring(text, "RGBA", True)
        glRasterPos2d(10, 45)
        glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Display Ollama's suggestion (if available)
        if self.ollama_suggestion:
            # Truncate if too long
            suggestion = self.ollama_suggestion[:40] + "..." if len(self.ollama_suggestion) > 40 else self.ollama_suggestion
            text = font.render(f"AI: {suggestion}", True, (200, 255, 200))
            text_data = pygame.image.tostring(text, "RGBA", True)
            glRasterPos2d(10, 70)
            glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Help text
        help_text = font.render("Use 1 hand to scale, 2 hands to rotate/switch shapes", True, (200, 200, 255))
        text_data = pygame.image.tostring(help_text, "RGBA", True)
        glRasterPos2d(self.config['display'][0] - help_text.get_width() - 10, self.config['display'][1] - 20)
        glDrawPixels(help_text.get_width(), help_text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def detect_gesture(self, hand_landmarks):
        """Detect hand gesture with improved accuracy"""
        # Get relevant landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate thumb-index distance
        thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                               (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        
        # Calculate average fingertip height relative to wrist
        avg_fingertip_y = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4
        fingers_raised = avg_fingertip_y < wrist.y - 0.1
        
        # Detect gestures
        if thumb_index_distance < 0.05:
            return "pinch"  # Pinching gesture (thumb and index finger close)
        elif fingers_raised and thumb_index_distance > 0.15:
            return "open"  # Open hand gesture (fingers up and spread)
        elif not fingers_raised:
            return "fist"  # Closed fist (fingers down)
        else:
            return "point"  # Default pointing gesture

    def hand_tracking_thread(self):
        """Process hand tracking in a separate thread"""
        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera_index'], cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open camera. Please check your camera connection.")
            self.stop_event.set()
            return

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from camera.")
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, self.config['camera_resolution'])

                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe Hands
                results = self.hands.process(rgb_frame)

                # Process results if hands detected
                if results.multi_hand_landmarks:
                    hand_data = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get the center of the hand (palm base)
                        palm_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        target_x = palm_base.x * 2 - 1  # Map to OpenGL coordinates (-1 to 1)
                        target_y = -(palm_base.y * 2 - 1)  # Invert y-axis for OpenGL
                        target_z = palm_base.z * 2  # Adjust depth scaling

                        # Detect gesture
                        gesture = self.detect_gesture(hand_landmarks)

                        # Add hand data to the list
                        hand_data.append((target_x, target_y, target_z, gesture))

                        # Draw landmarks on frame for debugging
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Put hand data in the queue
                    self.hand_data_queue.put(hand_data)
                
                # Display the frame with hand tracking
                cv2.imshow('Hand Tracking', frame)
                
                # Exit on 'Esc' key press
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()
                    break
                
                # Limit processing rate
                time.sleep(1/30)  # Cap at 30 FPS for hand tracking
                
        except Exception as e:
            print(f"Error in hand tracking thread: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def ask_ollama(self, prompt):
        """Interact with Ollama AI with improved error handling and timeout"""
        if not prompt:
            return None
            
        try:
            # Send request with timeout
            response = requests.post(
                self.config['ollama_endpoint'],
                json={"model": self.config['ollama_model'], "prompt": prompt},
                timeout=2.0  # 2 second timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response from AI")
            else:
                return f"Error: {response.status_code}"
        except requests.exceptions.Timeout:
            return "AI response timed out"
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama service"
        except Exception as e:
            return f"Error: {str(e)[:30]}"

    def run(self):
        """Main application loop"""
        # Initialize pygame and OpenGL
        self.init_pygame()
        
        # Start hand tracking thread
        tracking_thread = Thread(target=self.hand_tracking_thread, daemon=True)
        tracking_thread.start()
        
        last_ollama_time = 0
        last_shape_change_time = 0
        
        try:
            # Main application loop
            while not self.stop_event.is_set():
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop_event.set()
                    elif event.type == pygame.KEYDOWN:
                        # Handle keyboard inputs
                        if event.key == pygame.K_ESCAPE:
                            self.stop_event.set()
                        elif event.key == pygame.K_RIGHT:
                            self.current_shape_index = (self.current_shape_index + 1) % len(self.shapes)
                        elif event.key == pygame.K_LEFT:
                            self.current_shape_index = (self.current_shape_index - 1) % len(self.shapes)
                        elif event.key == pygame.K_UP:
                            self.object_scale = min(2.0, self.object_scale + 0.1)
                        elif event.key == pygame.K_DOWN:
                            self.object_scale = max(0.5, self.object_scale - 0.1)

                # Process hand tracking data
                if not self.hand_data_queue.empty():
                    hand_data = self.hand_data_queue.get()
                    current_time = time.time()
                    
                    if len(hand_data) == 1:
                        # One hand detected
                        self.target_x, self.target_y, self.target_z, self.gesture = hand_data[0]
                        
                        # Apply gesture effects
                        if self.gesture == "pinch":
                            self.object_scale = max(0.5, self.object_scale - 0.02)  # Make smaller
                        elif self.gesture == "open":
                            self.object_scale = min(2.0, self.object_scale + 0.02)  # Make bigger
                        elif self.gesture == "fist":
                            self.object_rotation[1] += 2  # Rotate
                        
                    elif len(hand_data) == 2:
                        # Two hands detected
                        _, _, _, gesture1 = hand_data[0]
                        _, _, _, gesture2 = hand_data[1]
                        
                        # Update gesture for display
                        self.gesture = f"{gesture1}+{gesture2}"
                        
                        if gesture1 == "open" and gesture2 == "open":
                            # Rotate the object faster
                            self.object_rotation[1] += 4
                        elif (gesture1 == "pinch" and gesture2 == "pinch" and 
                              current_time - last_shape_change_time > 1.0):  # 1 second cooldown
                            # Cycle through shapes
                            self.current_shape_index = (self.current_shape_index + 1) % len(self.shapes)
                            last_shape_change_time = current_time

                # Ask Ollama for suggestions (throttled to once per 3 seconds)
                current_time = time.time()
                if self.gesture and current_time - last_ollama_time > 3.0:
                    self.ollama_suggestion = self.ask_ollama(
                        f"Interpret this gesture for a 3D object manipulation program: {self.gesture}"
                    )
                    last_ollama_time = current_time

                # Clear the screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Setup lighting
                glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)

                # Draw the floor
                self.draw_floor()

                # Draw the object with transformations
                glPushMatrix()
                glTranslatef(self.target_x, self.target_y, self.target_z)
                glRotatef(self.object_rotation[0], 1, 0, 0)
                glRotatef(self.object_rotation[1], 0, 1, 0)
                glRotatef(self.object_rotation[2], 0, 0, 1)
                glScalef(self.object_scale, self.object_scale, self.object_scale)
                
                # Draw the current shape using function mapping
                current_shape = self.shapes[self.current_shape_index]
                self.shape_functions[current_shape]()
                
                glPopMatrix()

                # Draw the HUD
                self.draw_hud()

                # Update the display
                pygame.display.flip()
                self.clock.tick(self.config['frame_rate'])
                
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Ensure clean shutdown
            self.stop_event.set()
            tracking_thread.join(timeout=1.0)
            pygame.quit()
            cv2.destroyAllWindows()
            print("Application closed successfully")

def main():
    """Main entry point for the application"""
    try:
        # Create and run the application
        app = HandGestureApp()
        app.run()
    except KeyboardInterrupt:
        print("Application terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Ensure OpenGL and pygame are properly shut down
        pygame.quit()
        cv2.destroyAllWindows()
        print("Application resources released")

if __name__ == "__main__":
    main()