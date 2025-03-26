import sys
import importlib.util

def check_library(library_name):
    try:
        spec = importlib.util.find_spec(library_name)
        if spec is not None:
            print(f"{library_name} is installed")
            module = __import__(library_name)
            print(f"Version: {module.__version__ if hasattr(module, '__version__') else 'Unknown'}")
        else:
            print(f"{library_name} is NOT installed")
    except Exception as e:
        print(f"Error checking {library_name}: {e}")

def main():
    print("Python Version:", sys.version)
    print("\nChecking Libraries:")
    libraries_to_check = [
        'tensorflow', 
        'keras', 
        'numpy', 
        'cv2'
    ]
    
    for lib in libraries_to_check:
        check_library(lib)

if __name__ == "__main__":
    main()