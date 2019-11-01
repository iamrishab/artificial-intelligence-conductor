import subprocess
import os
import config

# p0 = subprocess.Popen(["python", "C:\\Users\\negis\\Downloads\\ppm-3\\speak_app.py"], stdout=subprocess.PIPE)
# p1 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'can_flask.py')], stdout=subprocess.PIPE)
p2 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'photo_sync.py')], stdout=subprocess.PIPE)
if config.MAIN_TYPE == 'API':
    if config.FETCH_DATA:
        p3 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'fetch_data.py')], stdout=subprocess.PIPE)
    p4 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'main_api.py')], stdout=subprocess.PIPE)
    print(p4.communicate())
elif config.MAIN_TYPE == 'EDGE':
    p3 = subprocess.Popen(["python", os.path.join(os.getcwd(), 'main.py')], stdout=subprocess.PIPE)
    print(p3.communicate())
