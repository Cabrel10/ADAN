import numpy as np

# Simulate the values from logs
# Values=[0.06666667014360428, -0.06666667014360428, 0.20000000298023224, -0.3333333432674408]
v1 = np.array([0.06666667014360428], dtype=np.float32)
v2 = np.array([-0.06666667014360428], dtype=np.float32)
v3 = np.array([0.20000000298023224], dtype=np.float32)
v4 = np.array([-0.3333333432674408], dtype=np.float32)

vals = [v1, v2, v3, v4]

print(f"Vals: {vals}")
print(f"Types: {[type(v) for v in vals]}")
print(f"Shapes: {[v.shape for v in vals]}")

median = np.median(vals)
print(f"Median: {median}")

# Check if it matches -1.0
if median == -1.0:
    print("REPRODUCED: Median is -1.0")
else:
    print(f"NOT REPRODUCED: Median is {median}")
