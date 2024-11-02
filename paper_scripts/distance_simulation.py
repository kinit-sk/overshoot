
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    
    
    
def visualize_path(points, title: str, margin: int = 3, resolution: int = 100):    
    plt.plot(points[:, 0], points[:, 1], '-or')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def foo(points, overshoot: float, momentum: float, window: int = 50):
    distances = []
    for i in range(1, points.shape[0] - window):
        current_momentum = points[i] - points[i-1]
        overshoot_point = points[i] + overshoot * current_momentum
        d = 0
        for j in range(i, min(i+window, points.shape[0])):
            d += np.linalg.norm(points[j] - overshoot_point) * momentum ** (j - i)
        distances.append(d)
    return np.mean(distances)

n_dimension = 20
n_points = 30000

# position = np.zeros(n_dimension)
cosine_sims = np.zeros((n_points))
model = np.zeros(n_dimension)
points = np.zeros((n_points, n_dimension))
momentum = 0

final_results = []
last_best_overshoot = 0
for m in tqdm([0.7, 0.73, 0.76, 0.79, 0.81, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99]):
    for i in range(n_points):
        g = np.random.randn(n_dimension)
        g /= np.linalg.norm(g) # norm to take equal steps
        new_momentum = m * momentum + g
        if i > 0:
            cosine_sims[i] = cosine_similarity(momentum, new_momentum)
            
        momentum = new_momentum
        model = np.add(model, momentum)
        points[i] = model
        


    print("==============================")
    print(f"Momentum: {m}")
    if int(m // 0.1) < 8:
        step_size = 1
    elif int(m // 0.1) == 8:
        step_size = 0.5
    elif int(m // 0.1) == 9:
        step_size = 0.2
        
    index = 0
    min_d = None
    while True: 
        d = foo(points, last_best_overshoot + index * step_size, momentum=m)
        print(f"Overshoot: {last_best_overshoot + index * step_size}, Distance: {d}")
        if index == 0:
            min_d = d
        elif d < min_d:
            min_d = d
        else:
            last_best_overshoot += (index - 1) * step_size # Last overshoot var the best
            break
        index += 1

    print(f"New points: {(m, last_best_overshoot)}")
    final_results.append((m, last_best_overshoot))

print("=====================================")
print(final_results)
    