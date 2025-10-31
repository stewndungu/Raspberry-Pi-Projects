import cv2
import face_recognition
import threading
import queue
from picamera2 import Picamera2
import time
import pickle
import numpy as np

#queue for processing the frame
frame_queue = queue.Queue()
result_queue = queue.Queue()
frame_count=0
start_time= time.time()
fps=0

'''
# Load a sample picture and learn how to recognize it
known_image = face_recognition.load_image_file("stewart.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Create an array of known face encodings and names
known_face_encodings = [known_encoding]
known_face_names = ["Stewart Ndung'u"]
'''

data = None
known_face_encodings= None
Known_face_names = None
print("[INFO] loading encodings...")
with open("encodings5.pickle", "rb") as f:
    data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

video_capture = Picamera2()
video_capture.configure(video_capture.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
video_capture.start()

face_locations, face_names = [],[]
def face_rec():
    global face_locations,face_names
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        if frame_count % 5 == 0:
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations,model='hog')
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # if there are no other names
                '''
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                '''
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        
        result_queue.put((face_locations, face_names))
    

thread = threading.Thread(target=face_rec, daemon=True)
thread.start()


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    #frame_count=frame_fps
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps



while True:
    frame = video_capture.capture_array()
    
    current_fps= calculate_fps()
    # if the queue is empty, queue the intial frame
    if frame_queue.empty():
        frame_queue.put(frame)
    
    # if the resutt_queue is empty, (usually after the frames have been processed in the queue using the face_rec)
    if not result_queue.empty():
        face_locations, face_names = result_queue.get()
      
    # put the names and boxes around it
    for (top,right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        if name == "unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
            
        cv2.putText(frame, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video Capture",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#clean up
frame_queue.put(None)
video_capture.release()
cv2.destroyAllwindows()