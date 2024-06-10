import subprocess

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        print(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    scripts = ['script-BKT-Timer.py', 'script-DLX-Timer.py', 'script-DLX-CPP-Timer.py']

    for script in scripts:
        run_script(script)