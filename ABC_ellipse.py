import numpy as np
from PIL import Image
import math
import random
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import time

# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image_from_device():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not file_path:
        print("No file selected.")
        return None

    image = Image.open(file_path).convert('L')
    print("Loaded image:", file_path)
    return image

# ==========================================
# STEP 2: EXTRACT EDGE POINTS USING CANNY
# ==========================================

def extract_points_from_image(image, low_threshold=50, high_threshold=150):
    """
    Extract edge points using Canny edge detection.

    low_threshold  - lower hysteresis threshold
    high_threshold - upper hysteresis threshold
    """

    # Convert PIL image to numpy array
    img = np.array(image)

    # Apply gaussian blur to reduce noise
    img = np.array(image)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(img, low_threshold, high_threshold)

    # Get image dimensions
    height, width = edges.shape

    # Extract coordinates of edge pixels
    y_coords, x_coords = np.where(edges > 0)

    points = list(zip(x_coords, y_coords))

    print("Number of extracted points:", len(points))
    print("Unique pixel values in original image:", np.unique(img))
    print("Unique pixel values in edge image:", np.unique(edges))

    return points, width, height


# ==========================================================
# FITNESS FUNCTION
# ==========================================================

def compute_fitness(ellipse, points):
    cx, cy, a, b, theta = ellipse

    if a <= 0 or b <= 0:
        return 0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    total_error = 0

    for (x, y) in points:
        dx = x - cx
        dy = y - cy

        x_r = dx * cos_t + dy * sin_t
        y_r = -dx * sin_t + dy * cos_t

        E = (x_r**2)/(a**2) + (y_r**2)/(b**2)
        error = abs(E - 1)

        total_error += error 

    avg_error = total_error / len(points)
    return 1.0 / (avg_error + 1e-6)

# ==========================================================
# INITIALIZE BEES
# ==========================================================

def initialize_bees(num_bees, width, height, points):
    bees = []
    fitness_values = []

    for _ in range(num_bees):
        cx = random.uniform(0, width)
        cy = random.uniform(0, height)
        a = random.uniform(10, width/2)
        b = random.uniform(10, height/2)
        theta = random.uniform(0, math.pi)

        ellipse = [cx, cy, a, b, theta]
        fitness = compute_fitness(ellipse, points)

        bees.append(ellipse)
        fitness_values.append(fitness)

    return bees, fitness_values

# ==========================================================
# EMPLOYED PHASE
# ==========================================================

def employed_bee_phase(bees, fitness_values, trial_counters,
                       points, width, height):

    num_bees = len(bees)

    for i in range(num_bees):

        k = random.randint(0, num_bees - 1)
        while k == i:
            k = random.randint(0, num_bees - 1)

        new_bee = bees[i].copy()
        param_index = random.randint(0, 4)
        phi = random.uniform(-1, 1)

        new_bee[param_index] = (
            bees[i][param_index]
            + phi * (bees[i][param_index] - bees[k][param_index])
        )

        # clamp
        new_bee[0] = max(0, min(width, new_bee[0]))
        new_bee[1] = max(0, min(height, new_bee[1]))
        new_bee[2] = max(5, min(width/2, new_bee[2]))
        new_bee[3] = max(5, min(height/2, new_bee[3]))
        new_bee[4] = max(0, min(math.pi, new_bee[4]))

        new_fitness = compute_fitness(new_bee, points)

        if new_fitness > fitness_values[i]:
            bees[i] = new_bee
            fitness_values[i] = new_fitness
            trial_counters[i] = 0
        else:
            trial_counters[i] += 1

    return bees, fitness_values

# ==========================================================
# ONLOOKER PHASE
# ==========================================================

def onlooker_bee_phase(bees, fitness_values, trial_counters,
                       points, width, height):

    num_bees = len(bees)
    total_fitness = sum(fitness_values)

    probabilities = [f / total_fitness for f in fitness_values]

    cumulative = []
    cumulative_sum = 0
    for p in probabilities:
        cumulative_sum += p
        cumulative.append(cumulative_sum)

    for _ in range(num_bees):

        r = random.uniform(0, 1)

        selected_index = 0
        for i in range(num_bees):
            if r <= cumulative[i]:
                selected_index = i
                break

        k = random.randint(0, num_bees - 1)
        while k == selected_index:
            k = random.randint(0, num_bees - 1)

        new_bee = bees[selected_index].copy()
        param_index = random.randint(0, 4)
        phi = random.uniform(-1, 1)

        new_bee[param_index] = (
            bees[selected_index][param_index]
            + phi * (bees[selected_index][param_index]
                     - bees[k][param_index])
        )

        new_bee[0] = max(0, min(width, new_bee[0]))
        new_bee[1] = max(0, min(height, new_bee[1]))
        new_bee[2] = max(5, min(width/2, new_bee[2]))
        new_bee[3] = max(5, min(height/2, new_bee[3]))
        new_bee[4] = max(0, min(math.pi, new_bee[4]))

        new_fitness = compute_fitness(new_bee, points)

        if new_fitness > fitness_values[selected_index]:
            bees[selected_index] = new_bee
            fitness_values[selected_index] = new_fitness
            trial_counters[selected_index] = 0
        else:
            trial_counters[selected_index] += 1

    return bees, fitness_values

# ==========================================================
# SCOUT PHASE
# ==========================================================

def scout_bee_phase(bees, fitness_values, trial_counters,
                    limit, width, height, points):

    for i in range(len(bees)):

        if trial_counters[i] >= limit:

            cx = random.uniform(0, width)
            cy = random.uniform(0, height)
            a = random.uniform(10, width/2)
            b = random.uniform(10, height/2)
            theta = random.uniform(0, math.pi)

            new_bee = [cx, cy, a, b, theta]

            bees[i] = new_bee
            fitness_values[i] = compute_fitness(new_bee, points)
            trial_counters[i] = 0

    return bees, fitness_values, trial_counters

# ==========================================================
# MAIN PROGRAM
# ==========================================================

image = load_image_from_device()

if image is None:
    exit()

points, width, height = extract_points_from_image(image, 30, 150)


num_bees = 40
limit = 25
max_iterations = 200

bees, fitness_values = initialize_bees(num_bees, width, height, points)
trial_counters = [0] * num_bees

best_index = fitness_values.index(max(fitness_values))
global_best = bees[best_index]
global_best_fitness = fitness_values[best_index]

fitness_history = []
start_time = time.time()

for iteration in range(max_iterations):

    bees, fitness_values = employed_bee_phase(
        bees, fitness_values, trial_counters,
        points, width, height
    )

    bees, fitness_values = onlooker_bee_phase(
        bees, fitness_values, trial_counters,
        points, width, height
    )

    bees, fitness_values, trial_counters = scout_bee_phase(
        bees, fitness_values, trial_counters,
        limit, width, height, points
    )

    current_best_index = fitness_values.index(max(fitness_values))
    current_best_fitness = fitness_values[current_best_index]

    if current_best_fitness > global_best_fitness:
        global_best = bees[current_best_index]
        global_best_fitness = current_best_fitness

    fitness_history.append(global_best_fitness)
    print("Iteration:", iteration,
          "Best Fitness:", global_best_fitness)
    
end_time = time.time()
total_runtime = end_time - start_time

print("Total Runtime:", total_runtime, "seconds")
print("Time per iteration:", total_runtime / max_iterations)
avg_error = 1.0 / global_best_fitness
print("Final Average Ellipse Residual Error:", avg_error)



# ==========================================================
# DRAW RESULT
# ==========================================================

cx, cy, a, b, theta = global_best

t = np.linspace(0, 2*np.pi, 200)

x_ellipse = cx + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
y_ellipse = cy + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)

plt.figure(figsize=(6,6))
plt.imshow(image, cmap='gray')

x_points = [p[0] for p in points]
y_points = [p[1] for p in points]

plt.scatter(x_points, y_points, color='red', marker='x', s=10)
plt.plot(x_ellipse, y_ellipse, color='blue', linewidth=2)

plt.title("Detected Ellipse")
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.plot(fitness_history)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("ABC Convergence Curve")
plt.grid()
plt.show()

