# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
model= tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		img=cv2.resize(frame,(224,224))

		
		# expand the dimensions
		img=np.expand_dims(img,axis=0)
		
		# normalize it before feeding to the model
		img=img/255
		
		# get predictions from the model
		predictions=model.predict(frame)
		print("Prediction:",predictions)
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)
		rock=int(predictions[0][0]*100)
		paper=int(predictions[0][1]*100)
		scissor=int(predictions[0][2]*100)

		print(f"rock: {rock}%, paper: {paper}%, scissor: {scissor}%")


		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
