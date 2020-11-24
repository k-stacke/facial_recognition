#! /usr/bin/python

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

def main(args):
	print("[INFO] start processing faces...")
	image_paths = list(paths.list_images(args.dataset_folder))
	
	# initialize the list of known encodings and known names
	known_encodings = []
	known_names = []

	# loop over the image paths
	for (i, image_path) in enumerate(image_paths):
		# extract the person name from the image path
		print(f"[INFO] processing image {i + 1}/{len(image_paths)}")
		name = image_path.split(os.path.sep)[-2]

		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		image = cv2.imread(image_path)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb, model="hog")

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			known_encodings.append(encoding)
			known_names.append(name)

	# dump the facial encodings + names to disk
	print("[INFO] serializing encodings...")
	data = {"encodings": known_encodings, "names": known_names}
	f = open("encodings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-folder', type=str, default='./dataset', 
						help='folder where images are stored')
	args = parser.parse_args()
	main(args)
	
