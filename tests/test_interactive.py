# %%
import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

# %%
# Test if packages are available
try:
    import ipykernel
    print(f"ipykernel version: {ipykernel.__version__}")
except ImportError:
    print("ipykernel not found!")

# %%
print("Interactive window test successful!")
