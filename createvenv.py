import subprocess
import platform

#note script doesn't work just pasta

mac_commands = [
    "python3 -m venv venv",
    "source venv/bin/activate",
    "pip install -r requirements.txt"
]

windows_commands = [
    "python -m venv venv",
    ".\venv\Scripts\activate",
    "pip install -r requirements.txt"
]

def run_commands(commands):
    for command in commands:
        subprocess.run(command.split())
        
if __name__ == "__main__":

    if platform.system() == 'Darwin':
        run_commands(mac_commands)
        print("source venv/bin/activate to start env\ndeactivate to exit")
    
    if platform.system() == 'Windows':
        run_commands(windows_commands)
        print(".\venv\Scripts\activate to start env\ndeactivate to exit")

# # Step 1: Create a virtual environment
# python -m venv venv

# # Step 2: Activate the virtual environment
# source venv/bin/activate   # On macOS/Linux
# # or
# .\venv\Scripts\activate    # On Windows

# # Step 3: Install dependencies
# pip install -r requirements.txt

# Step 4: Deactivate when done
# deactivate