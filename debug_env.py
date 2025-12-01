import sys
import os
print(f"Executable: {sys.executable}")
print(f"Version: {sys.version}")
print("Path:")
for p in sys.path:
    print(f"  {p}")

try:
    import numpy
    print(f"Numpy file: {numpy.__file__}")
    print(f"Numpy version: {numpy.__version__}")
except ImportError as e:
    print(f"Numpy import failed: {e}")
