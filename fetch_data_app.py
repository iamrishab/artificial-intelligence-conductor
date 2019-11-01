import subprocess
import os

p1 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'fetch_data.py')], stdout=subprocess.PIPE)
print(p1.communicate())

