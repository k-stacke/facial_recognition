#! /usr/bin/python

# import the necessary packages
import os
import time
import random
import argparse
import pickle
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition

def main(args):
	random.seed = 2
	input_file_no = args.sentences_stop
	input_file_ok = args.sentences_ok

	ok_person = args.name_exception

	sentences_ok = []
	sentences_no = []

	with open(input_file_no) as f:
		for line in f:
			sentences_no.append(line)
	with open(input_file_ok) as f:
		for line in f:
			sentences_ok.append(line)

	#Initialize 'currentname' to trigger only when a new person is identified.
	currentname = "unknown"
	#Determine faces from encodings.pickle file model created from train_model.py
	encodingsP = args.encodings
	#use this xml file
	cascade = args.haar_encodings

	# load the known faces and embeddings along with OpenCV's Haar
	# cascade for face detection
	print("[INFO] loading encodings + face detector...")
	data = pickle.loads(open(encodingsP, "rb").read())
	detector = cv2.CascadeClassifier(cascade)

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# start the FPS counter
	fps = FPS().start()

	frames_since_detect = 0
	name = "Unknown" #if face is not recognized, then print Unknown
	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to 500px (to speedup processing)
		frame = vs.read()
		frame = imutils.resize(frame, width=500)
		
		# convert the input frame from (1) BGR to grayscale (for face
		# detection) and (2) from BGR to RGB (for face recognition)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# detect faces in the grayscale frame
		rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		# OpenCV returns bounding box coordinates in (x, y, w, h) order
		# but we need them in (top, right, bottom, left) order, so we
		# need to do a bit of reordering
		boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

		# compute the facial embeddings for each face bounding box
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		# loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding)
			

			# check to see if we have found a match
			if True in matches:
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}

				# loop over the matched indexes and maintain a count for
				# each recognized face face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
				name = max(counts, key=counts.get)
				
				#If someone in your dataset is identified, print their name on the screen
				if (currentname != name) or ((currentname == name) and frames_since_detect > 10):
					currentname = name
					print(currentname)
					print('ok person? ',currentname == ok_person) 
					
					# Say hi!
					sentences = sentences_ok if currentname == ok_person else sentences_no
					current_sentence = random.choice(sentences)
					current_sentence = current_sentence.replace("<name>", currentname)
					
					if args.voice == 'google':
						os.system(f'./speech.sh {current_sentence} ')
					elif args.voice == 'espeak':
						os.system(f'espeak-ng -ven+f4 "{current_sentence}" ')
					
					frames_since_detect = 0
					
			
			# update the list of names
			names.append(name)

		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
			# draw the predicted face name on the image - color is in BGR
			cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 225), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				.8, (0, 255, 255), 2)

		# display the image to our screen
		cv2.imshow("Facial Recognition is Running", frame)
		key = cv2.waitKey(1) & 0xFF

		# quit when 'q' key is pressed
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()
		frames_since_detect += 1

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--encodings', type=str, default='./encodings.pickle',
						help='path to encodings of images')
	parser.add_argument('--haar-encodings', type=str, default='./haarcascade_frontalface_default.xml',
						help='path to pre-trained haar encodings')
	parser.add_argument('--sentences-ok', type=str, default='./sentences_ok.txt', 
						help='Text file containing sentences when passage is okay')
	parser.add_argument('--sentences-stop', type=str, default='./sentences_stop.txt', 
						help='Text file containing sentences when passage is denied')
	parser.add_argument('--name-exception', type=str, default='', 
						help='names of person for which passage is okay')
	parser.add_argument('--voice', type=str, default='google', choices=['google', 'espeak'],
						help='choose between "google" (requires internet access) and "espeak"')
	args = parser.parse_args()
	main(args)
