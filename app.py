from flask import Flask, jsonify, request
from flask_cors import CORS
from deepface import DeepFace

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

bucket_url = 'https://objectstorage.us-chicago-1.oraclecloud.com/p/j8d1ZFSVLZnXbJM07BW4ogJg-TEyu9uwb0or5vGoBPSRgDbyI2aj7lMynpBZ5rMw/n/axnwnsavbb9n/b/vision_service/o/'
# bucket_url = 'https://objectstorage.us-phoenix-1.oraclecloud.com/p/x0yMOkrLivMniq431TyH4eeVwlaIggaC1f8oz_KA9EmcOFyaogsZS8HaLO1W3OQB/n/axnwnsavbb9n/b/oda-vision/o/'

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/refresh', methods=['POST'])
def refresh():
	try:
		r = requests.get(bucket_url)
		objects = r.json()['objects']
	except Exception as requestError:
		return jsonify({"RequestError": str(requestError)}), 400

	for object in objects:
		try:
			bucketImage = urllib.request.urlopen(bucket_url + object['name'].replace(' ', '%20'))
			arr = np.asarray(bytearray(bucketImage.read()), dtype=np.uint8)
			img = cv2.imdecode(arr, -1)
			rgb_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			img_encoding = face_recognition.face_encodings(rgb_img2)[0]
			# img_encoding = DeepFace.represent(img, model_name='Facenet')[0]["embedding"]

			known_face_encodings.append(img_encoding)
			known_face_names.append(object['name'])
		except Exception as e:
			return jsonify({"EncodingError": str(e)}), 400

	try:
		np.save('known_face_encodings.npy', known_face_encodings)
	except Exception as faceEncodingfileError:
		return jsonify({"FaceEncodingFileError": str(faceEncodingfileError)}), 400

	try:
		with open("known_face_names.txt", "w") as txt_file:
			for line in known_face_names:
				txt_file.write(line + "\n")
	except Exception as faceNamefileError:
		return jsonify({"FaceNameFileError": str(faceNamefileError)}), 400

	response = jsonify({"status": "completed"})
	return response

@app.route('/analyze', methods=['POST'])
def analyze():
	try:
		image_base64 = request.json['image']

		try:
			known_face_encodings = np.load('known_face_encodings.npy')
		except Exception as loadFileError:
			return jsonify({"loadEncodingsFileError": str(loadFileError)}), 400

		try:
			with open("known_face_names.txt", 'r') as f:
				known_face_names = f.read().splitlines()
		except Exception as loadFileError:
			return jsonify({"loadNamesFileError": str(loadFileError)}), 400

		try:
			image_data = base64.b64decode(image_base64)
		except base64.binascii.Error:
			return jsonify({"Base64DecodeError": "Invalid base64 string"}), 400

		try:
			image = Image.open(io.BytesIO(image_data))
			image_np = np.array(image)
		except Exception as e:
			return jsonify({"ImageOpenError": str(e)}), 400

		# try:
		# 	# Use OpenCV to detect faces
		# 	gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
		# 	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		# 	faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

		# 	# Extract face locations
		# 	face_locations = []
		# 	for (x, y, w, h) in faces:
		# 		face_locations.append([y, x+w, y+h, x])  # Convert to top, right, bottom, left format

		# 	# # Use DeepFace to recognize faces in detected locations
		# 	face_names = []
		# 	face_names.append("Unknown")
		# 	# for face_location in face_locations:
		# 	# 	top, right, bottom, left = face_location
		# 	# 	face_image = image_np[top:bottom, left:right]

        #         # # Use DeepFace to find the best match
		# 		# try:
		# 		# 	result = DeepFace.verify(face_image, known_face_encodings, model_name='Facenet')
		# 		# 	if result['verified']:
		# 		# 		face_names.append(result['verified'])
		# 		# 	else:
		# 		# 		face_names.append("Unknown")
		# 		# except Exception as e:
		# 		# 	return jsonify({"DeepFaceVerifyError": str(e)}), 400
		# except Exception as e:
		# 	return jsonify({"FaceRecognitionError": str(e)}), 400

		# response = jsonify({"face_ids": face_names, "face_locations": face_locations})
		# return response

		try:
			face_locations = face_recognition.face_locations(image_np)
			face_encodings = face_recognition.face_encodings(image_np, face_locations)
		except Exception as e:
			return jsonify({"FaceEncodingError": str(e)}), 400

		face_names = []
		for face_encoding in face_encodings:
			try:
				# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"
			except Exception as e:
				return jsonify({"CompareFacesError": str(e)}), 400

			try:
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index]:
					name = known_face_names[best_match_index]
				face_names.append(name)
			except Exception as e:
				return jsonify({"FaceDistanceError": str(e)}), 400

		face_locations = np.array(face_locations)
		response = jsonify({"face_ids": face_names, "face_locations": face_locations.astype(int).tolist()})
		return response
	except Exception as genErr:
		return jsonify({"GeneralError": str(genErr)}), 400

if __name__ == '__main__':
	app.run(port=3000)
