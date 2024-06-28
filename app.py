from flask import Flask, jsonify, request
from flask_cors import CORS

import cv2
import face_recognition
import urllib.request
import numpy as np
import requests
import base64
from PIL import Image
import io

known_face_encodings = []
known_face_names = []

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# bucket_url = 'https://objectstorage.us-chicago-1.oraclecloud.com/p/j8d1ZFSVLZnXbJM07BW4ogJg-TEyu9uwb0or5vGoBPSRgDbyI2aj7lMynpBZ5rMw/n/axnwnsavbb9n/b/vision_service/o/'
bucket_url = 'https://objectstorage.us-phoenix-1.oraclecloud.com/p/x0yMOkrLivMniq431TyH4eeVwlaIggaC1f8oz_KA9EmcOFyaogsZS8HaLO1W3OQB/n/axnwnsavbb9n/b/oda-vision/o/'

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/refresh', methods=['POST'])
def refresh():	
	r = requests.get(bucket_url)
	objects = r.json()['objects']

	for object in objects:
		bucketImage = urllib.request.urlopen(bucket_url + object['name'].replace(' ', '%20'))
		arr = np.asarray(bytearray(bucketImage.read()), dtype=np.uint8)
		img2 = cv2.imdecode(arr, -1)
		# img2 = cv2.imread("images/Tayler Duncan.jpeg")
		rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
		img_encoding = face_recognition.face_encodings(rgb_img2)[0]

		known_face_encodings.append(img_encoding)
		known_face_names.append(object['name'])
		
	response = jsonify({"status": "completed"})
	return response

@app.route('/analyze', methods=['POST'])
def analyze():
	image_base64 = request.json['image']

	try:
		image_data = base64.b64decode(image_base64)
	except base64.binascii.Error as e:
		return jsonify({"error": "Invalid base64 string"}), 400	

	image = Image.open(io.BytesIO(image_data))
	image_np = np.array(image)

	face_locations = face_recognition.face_locations(image_np)
	face_encodings = face_recognition.face_encodings(image_np, face_locations)
		
	face_names = []
	for face_encoding in face_encodings:
	
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
		name = "Unknown"

		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)		
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]
		face_names.append(name)
			
	face_locations = np.array(face_locations)
	response = jsonify({"face_ids": face_names, "face_locations": face_locations.astype(int).tolist()})	
	return response

if __name__ == '__main__':
	
	app.run(port=3000)