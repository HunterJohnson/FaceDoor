# Hunter Johnson - 2018
# program runs as infinite loop until terminated


import RPi.GPIO as GPIO
from motion_detection import motion_detected
from face_detection import face_detected
from validate_entry import valid_entrant
import os

from gpiozero import Button, OutputDevice

relay = OutputDevice(17)



def main():
	relay.on() # lock door
	while(True):
		if(motion_detected()):
			if(face_detected()):
				print("Face Detected!")
				#or load all face pics in past 10s timeframe instead of getting latest
				img = getLatestFace()   # all detected faces are cropped to bounding box and saved as jpg
				if(valid_entrant(img)):
					print("Permission Granted!")
					unlockDoor()
				else:
					print("Access Denied")
			else:
				print("No Face Detected")
		else:
			print("No motion detected")

			
if __name__ == "__main__":
	main()


def getLatestFace(): # most recent face capture
	latest_file = max(allFilesUnder('/home/pi/Documents/frdoor_project/'), key=os.path.getmtime)
	return latest_file

def allFilesUnder(path):
	for cur_path, dirnames, filenames in os.walk(path):
		for filename in filenames:
			yield os.path.join(cur_path, filename)


def unlockDoor():
    relay.off()
    time.sleep(5) # 5 seconds until door re-locks, adjust as necessary 
    relay.on()

