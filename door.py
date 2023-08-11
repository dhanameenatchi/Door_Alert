import cv2
import time
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import dlib
from math import hypot
from twilio.rest import Client
import requests
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
# from PIL import ImageGrab
# path to get database images
path = 'FaceData'
images = []
classNames = []
myList = os.listdir(path)

# # Initialize variables
# start_time = 0
# duration = 0
# person_detected = False

print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#detect frontal face of person
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\virtual\shape_predictor_68_face_landmarks.dat")


# def timecalc():
#     Open the camera
#     cap = cv2.VideoCapture(0)
#
#
#
#     # Define the font and position for the duration text
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     pos = (20, 50)
#
#     while True:
#         # Read the frame from the camera
#         ret, frame = cap.read()
#
#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Detect people in the frame
#         detector = cv2.CascadeClassifier(r'D:\virtual\haarcascade_frontalface_default.xml')
#         people = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#         # If a person is detected
#         if len(people) > 0:
#             # If a person wasn't detected before
#             if not person_detected:
#                 # Set the start time
#                 start_time = time.time()
#                 person_detected = True
#         else:
#             # If a person was detected before
#             if person_detected:
#                 # Calculate the duration
#                 duration = time.time() - start_time
#                 person_detected = False
#
#         # Display the frame with the detection results
#         for (x, y, w, h) in people:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         # Draw the duration text on the frame
#         text = f"Duration of person: {duration:.2f} seconds"
#         cv2.putText(frame, text, pos, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#
#         # Exit if the user presses 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the camera and close the window
#     cap.release()
#     cv2.destroyAllWindows()
#
#     print(duration)
#     return duration
#  dura = timecalc()

#discord message generator for known persons
def knowndiscordmsg(name):
    contnt = name
    payload = {
        'content': f"{contnt} HAS COME TO YOUR DOOR"
    }
    header = {
        'Authorization': 'place discord authorization token here'
    }

    r = requests.post("place the discord channel url", data=payload, headers=header)


def unknowndiscord_msg():
    # discord message generator for unknown persons
    payload = {
        'content': "UNKNOWN USER CAME TO YOUR DOOR"
    }
    header = {
        'Authorization': 'place discord authorization token here'
    }

    r = requests.post("place the discord channel url", data=payload, headers=header)

def cellphn_alert():
    # discord message generator for unknown persons
    payload = {
        'content': "SOMEONE IS TRYING TO BYPASS YOUR DOOR ALERT SYSTEM BY SHOWING THE IMAGE IN A PHONE"
    }
    header = {
        'Authorization': 'place discord authorization token here'
    }

    r = requests.post("place the discord channel url", data=payload, headers=header)


#fn to calculate face distance from camera
def face_distance(img):
    #finding face coordinates
    img1, faces = detector1.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector1.findDistance(pointLeft, pointRight)
        W = 6.3

        # Finding distance
        f = 840
        d = (W * f) / w
        return d

def message1(name):
    account_sid = 'Twilio_sid'
    auth_token = 'Twilio_authtoken'
    client = Client(account_sid, auth_token)
    msg = name + ". has arrived to your door"
    message = client.messages.create(
        from_='Twilio phone number',
        body=msg,
        to='Receiver Phone Number'
    )

    print(message.sid)
def message2():
    account_sid = 'Twilio_sid'
    auth_token = 'Twilio_authtoken'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_='Twilio phone number',
        body='unknown person came to your door',
        to='Receiver phone number'
    )

    print(message.sid)



#calculating mid point for blink
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
font = cv2.FONT_HERSHEY_PLAIN
#fn to find blink count
def get_blinking_ratio(frame,eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

#fn to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        #converting images to rgb color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#fn for calculating the duration of the person in camera

def udoorlog():

    with open('udoorlog.txt', 'a+') as f:
        now = datetime.now()
        dtstring = now.strftime('%H:%M:%S  on %d-%m-%y ')
        f.writelines(f'\nUNKNOWN @,{dtstring}')


def doorlog(name):

    #OPENING doorlog.tx for marking doorlog
    with open('good.txt', 'a+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
        # if name not in nameList:
            now = datetime.now()
            #dtstring = now.strftime('%d-%m-%y %H:%M:%S')
            tod = date.today()
            dtstring = now.strftime('%H:%M:%S  on %d-%m-%y')
            f.writelines(f'\n{name},{dtstring}')


#fn to find mobile phone
def detect_phone(img):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    #print(classIds,bbox,)
    try:

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            pass

            #print(classNames[classId - 1])
            #cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            #cv2.putText(img, classNames1[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        #(0, 255, 0), 2)
            #cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        return classNames1[classId - 1]
    except AttributeError:
        #no person in camera run properly
        pass


#fn to count blinking
def blinking(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coun=0

    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio(frame,[36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio(frame,[42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 4.7:
            #global coun
            #cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            coun+= 1

    return coun

#FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr


encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print('Encoding Complete')
#starts capturing video from camera
cap = cv2.VideoCapture(0)
#count=0
classNames1 = []
classFile = "D:\\virtual\coco.names"
with open(classFile,'rt') as f:
    classNames1 = f.read().rstrip('\n').split('\n')
configPath = "D:\\virtual\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath ="D:\\virtual\\frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
detector1 = FaceMeshDetector(maxFaces=1)
sen=10

while True:



    #reading images
    success, img = cap.read()
    # img = captureScreen()
    detect = detect_phone(img)
    blink = blinking(img)
    fdist = face_distance(img)

    if detect == "person":
        # print(detect)
        # print(blink)
        # print(fdist)
        if fdist == None:
            # if no person in camera should not throw error
            pass
        elif fdist > 70:
            # resizing the frame image
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                # comparing faces to database
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)
                name = classNames[matchIndex].upper()
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # rectangle in the faces
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    # putting text in output frame
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    # calling doorlog to mark in csv file
                    if blink >= 1:
                        # print(detect)
                        # print(blink)
                        # print(fdist)
                        # knowndiscordmsg(name)
                        # doorlog(name)
                        print(name, "has arrived")
                        # calling message fn to send msg
                        message1(name)
                        # time.sleep(120)
                elif name != classNames[matchIndex]:
                    if blink >= 1:
                        # print(detect)
                        # print(blink)
                        # print(fdist)
                        # unknowndiscord_msg()
                        # message2()
                        # udoorlog()
                        print('UNKNOWN person has arrived')


    elif detect == "cell phone":
        print("****dont show cell phone****")
        cellphn_alert()
    cv2.imshow('Webcam', img)
    a = cv2.waitKey(1)
    # pressing q to exit
    if a == ord("q"):
        break
