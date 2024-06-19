import subprocess


# Script that runs the 3 scripts for the 3 modes of solving puzzles.
def runScript(scriptName):
    try:
        result = subprocess.run(['python', scriptName], check=True, capture_output=True, text=True)
        print(f"Output of {scriptName}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {scriptName}:\n{e.stderr}")


if __name__ == "__main__":
    scripts = ['script-BKT-Timer.py', 'script-DLX-Timer.py', 'script-DLX-CPP-Timer.py']

    for script in scripts:
        runScript(script)
