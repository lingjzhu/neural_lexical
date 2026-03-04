import subprocess
import os
import sys

# Path to the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(CURRENT_DIR, script_name)
    print(f"\n🚀 --- Running {script_name} ---")
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)

def main():
    scripts = [
        "cluster_analysis.py",
        "compare_clusters.py",
        "token_alignment_analysis.py",
        "activation_frequency_analysis.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n✨ --- All Clustered ColBERT Analysis Complete --- ✨")
    print(f"Check results in: {os.path.join(CURRENT_DIR, 'results')}")

if __name__ == "__main__":
    main()
