import cv2
import mediapipe as mp
import tkinter as tk
import time

# Maze dimensions
ROWS, COLS = 10, 10
CELL_SIZE = 50

# Maze grid (0 = path, 1 = wall)
maze = [
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
]

# Initialize Mediapipe for hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Function to detect the number of fingers raised
def fingers_up(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Finger tips
    # Thumb (special case)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Maze game class
class MazeGame:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE)
        self.canvas.pack()
        self.draw_maze()

        self.player_pos = [0, 0]  # Start position
        self.player = self.canvas.create_oval(
            5, 5, CELL_SIZE-5, CELL_SIZE-5, fill='blue'
        )

    def draw_maze(self):
        for row in range(ROWS):
            for col in range(COLS):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                if maze[row][col] == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

    def move_left(self):
        if self.player_pos[0] > 0 and maze[self.player_pos[1]][self.player_pos[0] - 1] == 0:
            self.player_pos[0] -= 1
            self.update_player_position()

    def move_right(self):
        if self.player_pos[0] < COLS - 1 and maze[self.player_pos[1]][self.player_pos[0] + 1] == 0:
            self.player_pos[0] += 1
            self.update_player_position()

    def move_up(self):
        if self.player_pos[1] > 0 and maze[self.player_pos[1] - 1][self.player_pos[0]] == 0:
            self.player_pos[1] -= 1
            self.update_player_position()

    def move_down(self):
        if self.player_pos[1] < ROWS - 1 and maze[self.player_pos[1] + 1][self.player_pos[0]] == 0:
            self.player_pos[1] += 1
            self.update_player_position()

    def update_player_position(self):
        x1 = self.player_pos[0] * CELL_SIZE + 5
        y1 = self.player_pos[1] * CELL_SIZE + 5
        x2 = x1 + CELL_SIZE - 10
        y2 = y1 + CELL_SIZE - 10
        self.canvas.coords(self.player, x1, y1, x2, y2)

# Initialize the game
root = tk.Tk()
root.title("Maze Runner with Hand Gestures")
game = MazeGame(root)

# Start the camera feed for hand gestures
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1)

# Time-based control to limit frame processing frequency
prev_time = time.time()
frame_delay = 0.1  # Delay in seconds (i.e., process every 0.1 seconds, or 10 FPS)

while True:
    success, img = cap.read()
    if not success:
        continue

    # Check if enough time has passed to process the next frame
    current_time = time.time()
    if current_time - prev_time > frame_delay:
        prev_time = current_time  # Update the last processing time
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                finger_states = fingers_up(handLms)
                total_fingers = sum(finger_states)

                # Map finger gestures to maze controls
                if total_fingers == 1:
                    game.move_up()  # Move up with 1 finger
                    print("up")
                elif total_fingers == 2:
                    game.move_right()  # Move right with 2 fingers
                    print("Right")
                elif total_fingers == 3:
                    game.move_down()  # Move down with 3 fingers
                    print("down")
                elif total_fingers == 4:
                    game.move_left()  # Move left with 4 fingers
                    print("left")

    # Display the camera feed with hand detection
    cv2.imshow("Hand Gesture Control", img)
    
    # Update the Tkinter window
    root.update()

    # Break if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
