import subprocess
import os

# p0 = subprocess.Popen(["python", "C:\\Users\\negis\\Downloads\\ppm-3\\speak_app.py"], stdout=subprocess.PIPE)
p1 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'can_flask.py')], stdout=subprocess.PIPE)
print(p1.communicate())
