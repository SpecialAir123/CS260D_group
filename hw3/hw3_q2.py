import numpy as np
from itertools import combinations

data = {
    1: np.array([0.0, 0.0, 1.5, 3.0, -0.05]),
    2: np.array([4.0, 6.0, 0.0, 0.0, 0.05]),
    3: np.array([6.0, 4.0, 0.0, 0.0, -0.05]),
    4: np.array([6.0, 6.0, 0.0, 0.0, -0.05]),
    5: np.array([8.0, 6.0, 0.0, 0.0, -0.05]),
    6: np.array([0.0, 0.0, 1.5, 4.5, -0.05]),
    7: np.array([6.0, 6.0, 0.0, 0.0, 0.05]),
    8: np.array([0.0, 0.0, 1.5, 6.0, 0.05]),
    9: np.array([6.0, 8.0, 0.0, 0.0, 0.05]),
    10: np.array([0.0, 0.0, 1.0, 4.5, 0.05]),
    11: np.array([0.0, 0.0, 2.0, 4.5, -0.05]),
    12: np.array([0.0, 0.0, 1.5, 4.5, 0.05])
}

def l1_norm(v1, v2):
    return np.sum(np.abs(v1 - v2))

def calculate_objective(selected, v0, v1, data):
    total_distance = 0
    selected_class0 = [x for x in selected if x in v0]
    non_selected_class0 = [x for x in v0 if x not in selected]

    for i in selected_class0:
        for j in non_selected_class0:
            total_distance += l1_norm(data[i], data[j])

    selected_class1 = [x for x in selected if x in v1]
    non_selected_class1 = [x for x in v1 if x not in selected]

    for i in selected_class1:
        for j in non_selected_class1:
            total_distance += l1_norm(data[i], data[j])

    return total_distance

v0 = [2, 3, 4, 5, 7, 9]
v1 = [1, 6, 8, 10, 11, 12]
class0_combos = list(combinations(v0, 2))
class1_combos = list(combinations(v1, 2))

min_obj = float('inf')
best_combo = None

for combo0 in class0_combos:
    for combo1 in class1_combos:
        current_selection = list(combo0) + list(combo1)
        obj = calculate_objective(current_selection, v0, v1, data)

        if obj < min_obj:
            min_obj = obj
            best_combo = current_selection

print(f"Best combination: {best_combo}")
print(f"Minimum objective value: {min_obj}")

# Verify the selected examples
print("\nSelected examples:")
for idx in best_combo:
    print(f"x{idx} = {data[idx]}, class = {'0' if idx in v0 else '1'}")