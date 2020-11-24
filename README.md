# Surverillence Camera with Raspberry Pi

This "surveillence camera" uses pre-trained haar features to find faces. 
The detected faces are compared agains dataset of known people. If a match is found,
the system will output warning statements, randomly from `sentences_stop.txt`. 
Use "\<name\>" as placeholder for the detected person. 

Example sentence:
`"Stop! I see you, <name>!"`

How to run:

1. `python3 headshot.py --name NAMEOFPERSON` run to capture 
images of people you want to be able to recognize. Repeat for all people.

2. `python3 encode_images.py` creates encodings of the captured images. 
These encodings will be used to compare agains when the system is live.

3. `python3 facial_req.py` use to run the application. If you want one of the detected 
persons to be allowed, add an excetion with the `--name-exception` flag. 
This person will be read sentences from `sentences_ok.txt` file. 



## Acknowledgments
This repo is greatly inpired by https://github.com/carolinedunn/facial_recognition,
and the tutorial https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/.
