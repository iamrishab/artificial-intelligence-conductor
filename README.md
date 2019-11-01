##Installing the packages
1.	Download the “ppm-can” repository from the bitbucket link.
2. Double click on the "install.bat" file.
3. This will install all the required packages for our application.

##Creating the application shortcut
1.	Download the “ppm-can” repository from the bitbucket link.
2.	Create a desktop shortcut of the “app.bat” file from the repository folder.
3.	Rename the shortcut to “ppm”.
4.	Change the application icon to the “app.ico” file located in images folder in the repository.
5.	Create a desktop shortcut of the “start_can.bat” file from the repository folder.
6.	Rename the shortcut to “can”.
7.	Change the application icon to the “can.ico” file located in images folder in the repository.

##Starting the application
1.	Double click on the icon on the desktop named “ppm” with PPM icon to start the application.
2.	The application takes 1 – 1.25 minutes to start. Please do not interrupt the application when its starting. 
Stopping the application

1.	At first click on the application screen.
2.	Turn on the Caps Lock by pressing the “Caps” key on the keyboard.
3.	Then Press “Q”.
4.	The application will quit.

##Assumptions/State change
* RFID 1/3 – START
* RFID 2/4 – STOP
* FR – Face Recognition
* GR – Gesture Recognition
1.	When vehicle will hit RFID 1, 2, 3, 4. It will stop automatically.
2.	When we get RFID-1 and RFID-3, FR/GR will start automatically (depends on the configuration of the application).
3.	When we get RFID-2 and RFID-4, application will reset and reach and idle state until the vehicle reaches any start RFID.
4.	When there is an obstacle in front of the vehicle, it will stop automatically.
5.	Manual/Remote control start -> palm-up state will be reached.
6.	Manual/Remote control stop -> thumbs-up state will be reached.
5.	When vehicle moves from RFID-4 to RFID-1 and RFID-2 to RFID-3, application will not do anything, it will be in idle state.
6.	At RFID -1 and RFID-3, before FR/GR if any obstacle is detected, it will be ignored by the application.
7.	When vehicle is in STOP condition after palm-up, do not do manual start of the vehicle until and unless it reaches Thumbs-up condition.
