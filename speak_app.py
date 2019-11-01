#!flask/bin/python
from flask import Flask, request, json
import win32com.client as wincl
import pythoncom, threading, time


def start():
    # Initialize
    pythoncom.CoInitialize()

    # Get instance
    xl = wincl.Dispatch('SAPI.SpVoice')

    # Create id
    xl_id = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, xl)

    # Pass the id to the new thread
    thread = threading.Thread(target=run_in_thread, kwargs={'xl_id': xl_id})
    thread.start()

    # Wait for child to finish
    # thread.join()

    return xl


def run_in_thread(xl_id):
    # Initialize
    pythoncom.CoInitialize()

    # Get instance from the id
    xl = wincl.Dispatch(
        pythoncom.CoGetInterfaceAndReleaseStream(xl_id, pythoncom.IID_IDispatch)
    )
    print('yes')
    time.sleep(5)


app = Flask(__name__)

speak = start()

stop_speak = False


@app.route('/', methods=['GET', 'POST'])
def index():
    if not stop_speak:
        response = request.get_data()
        data = response.decode('utf-8')
        data = data.replace('+', ' ')
        print(data)
        # speak.Speak(data['text'])
        speak.Speak(data.split('=')[1])
    # return "Hello, World!"


@app.route('/stop', methods=['GET', 'POST'])
def stop_speaking():
    print('stop called')
    stop_speak = True


if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)
