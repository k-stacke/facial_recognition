import os
import cv2
import argparse

def main(args):
    name = args.name.lower()
    folder = args.output_folder
    if not os.path.isdir(folder):
        print(f'Output folder not found, creating new folder at: {folder}')
        os.mkdir(folder)
    if not os.path.isdir(f'{folder}/{name}'):
        os.mkdir(f'{folder}/{name}')
        
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = f'{folder}/{name}/image_{img_counter}.jpg'
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of person to capture')
    parser.add_argument('--output-folder', type=str, default='./dataset', help='folder to store images')
    args = parser.parse_args()
    main(args)
