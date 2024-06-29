from flask import Flask, request, jsonify
from flask_cors import CORS
from IPython import get_ipython
import os
import subprocess
import base64
from pyngrok import ngrok
import random
import librosa
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from flask import jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import assemblyai as aai
aai.settings.api_key = os.getenv('AAI_API')


app = Flask(__name__)
CORS(app)

def check_grammar(input_text):
    apikey=os.getenv('FIREWORKS')
    import requests
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    payload = {
    "model": "accounts/fireworks/models/llama-v3-70b-instruct",
    "max_tokens": 1024,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
    "messages": [
        {
            "role": "user",
            "content": "The user inputs a text and prompt a percentage value of the grammar of the sentence. If it is incomplete or with errors reduce the marks based on that.output only marks as integer. If the total words is less than 6 then output 15"
        },
        {
            "role": "user", "content": input_text
        }
    ],
    "response_format": {"type": "text"}
}
    headers = {
    "Authorization": f"Bearer {apikey}",
    "Content-Type": "application/json"
}
    import json
    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)
    response_json = json.loads(response.text)

    content = response_json["choices"][0]["message"]["content"]
    return content

UPLOAD_FOLDER = '/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def upload_video(video_data):        
   

    if video_data:
        video_name = 'uploaded_video.webm'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        print(video_path)
        print(video_name)
        video_data.save(video_path)  # Save the file
        
        
        
        # return result
        return jsonify({'message': 'Video uploaded successfully'}), 200
    else:
        return jsonify({'error': 'No video data provided'}), 400
    

def extract_audio(input_file, output_file):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-vn",  # This option tells FFmpeg to ignore the video stream
        "-acodec", "mp3",  # Explicitly set the audio codec to mp3
        "-y", output_file,
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        result.check_returncode()  # Raise an error if the command failed
        print("Audio extraction successful.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")

def voice_extraction_main():
    input_file = '/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/uploaded_video.webm'
    output_file = '/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/output.mp3'  # Replace with the desired name for the output MP3 file

    extract_audio(input_file, output_file)
    

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the class indices
class_indices = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

def eye_tracking_method(video_path):
    cascade_classifiers = [
        cv2.CascadeClassifier("/Users/sagni/OneDrive/Desktop/AI_Interview/backend/eye_track/haarcascade_eye.xml"),
        cv2.CascadeClassifier("/Users/sagni/OneDrive/Desktop/AI_Interview/backend/eye_track/haarcascade_frontalface_default.xml")
    ]

    def process_frame(frame, eye_detector):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(gray, 1.3, 5)
        eye_centers = [(eye[0] + eye[2] // 2, eye[1] + eye[3] // 2) for eye in eyes]
        if len(eye_centers) >= 2:
            eye_distance = np.linalg.norm(np.array(eye_centers[0]) - np.array(eye_centers[1]))
            gaze_percentage = eye_distance / (frame.shape[1] // 2)
        else:
            gaze_percentage = -1
        return gaze_percentage

    cap = cv2.VideoCapture(video_path)
    gaze_percentages = []

    with ThreadPoolExecutor(max_workers=len(cascade_classifiers)) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = list(executor.map(process_frame, [frame] * len(cascade_classifiers), cascade_classifiers))
            average_gaze = sum(results) / len(results)
            gaze_percentages.append(average_gaze)

    cap.release()
    if gaze_percentages:
        final_output = (sum(gaze_percentages) / len(gaze_percentages)) * 99 * -1
        print(f"Average Gaze Percentage: {final_output:.2f}")
        return final_output
    else:
        print("No gaze data available.")
        return 1

def classify_emotion(predictions, class_indices):
    happiness_index = class_indices['happy']
    happiness_percentage = predictions[0][happiness_index] * 100
    return happiness_percentage

def check_happiness(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        print(f"TensorFlow version: {tf.__version__}")
        emotion_model = load_model('/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/modelaccurate.h5')
        #opt = Adam(learning_rate=0.0001)
        #emotion_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
      
        emotion_model.summary()
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return 1

        total_happiness_percentage = 0
        frame_count = 0
        gaze_percentage = eye_tracking_method(video_path)
        if gaze_percentage < 0 or gaze_percentage == 100:
            print(gaze_percentage)
            return 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if gaze_percentage > 0:
                if np.mean(frame) < 10:
                    return 1
                else:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_frame = cv2.resize(gray_frame, (56, 56))
                    resized_frame = resized_frame.astype('float32') / 255.0
                    input_image = np.expand_dims(resized_frame, axis=0)
                    input_image = np.expand_dims(input_image, axis=-1)
                    try:
                        predictions = emotion_model.predict(input_image)
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")
                        return 1
                    happiness_percentage = classify_emotion(predictions, class_indices)
                    total_happiness_percentage += float(happiness_percentage)
                    frame_count += 1

        cap.release()

        if frame_count > 0:
            average_happiness = total_happiness_percentage / frame_count
            print(f'Average Happiness Percentage: {average_happiness:.2f}%')
            print(gaze_percentage)
            if gaze_percentage > 99:
                return 1
            average_result = (average_happiness + float(gaze_percentage)) / 2
            if average_result > 99:
                return 1
            return average_result
        else:
            print("No frames available for processing.")
            return 1

    except Exception as e:
        print(f"Error in check_happiness model: {str(e)}")
        return 1




'''def eye_tracking_method(video_path):
    # eye_detector = cv2.CascadeClassifier("/home/jegathees5555/Documents/recruitz/backend/eye_track/haarcascade_eye.xml")
    cascade_classifiers = [
        cv2.CascadeClassifier("/Users/sagni/OneDrive/Desktop/AI_Interview/backend/eye_track/haarcascade_eye.xml"),
        cv2.CascadeClassifier("/Users/sagni/OneDrive/Desktop/AI_Interview/backend/eye_track/haarcascade_frontalface_default.xml")
        # Add your second Haar Cascade classifier here if needed
    ]

    def process_frame(frame, eye_detector):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the eyes in the frame
        eyes = eye_detector.detectMultiScale(gray, 1.3, 5)

        # Find the center of the eyes
        eye_centers = []
        for eye in eyes:
            eye_center = (eye[0] + eye[2] // 2, eye[1] + eye[3] // 2)
            eye_centers.append(eye_center)

        if len(eye_centers) >= 2:
            # Calculate the distance between the eyes
            eye_distance = np.linalg.norm(np.array(eye_centers[0]) - np.array(eye_centers[1]))

            # Calculate the percentage of how much the user is looking into the camera
            gaze_percentage = eye_distance / (frame.shape[1] // 2)
        else:
            gaze_percentage = -1  # Default value if eyes are not detected

        return gaze_percentage

    cap = cv2.VideoCapture(video_path)
    gaze_percentages = []

    with ThreadPoolExecutor(max_workers=len(cascade_classifiers)) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame with different Haar Cascade classifiers in parallel
            results = list(executor.map(process_frame, [frame] * len(cascade_classifiers), cascade_classifiers))

            # Use the results as needed (e.g., take the average)
            average_gaze = sum(results) / len(results)
            gaze_percentages.append(average_gaze)

    cap.release()

    if gaze_percentages:
        final_output = (sum(gaze_percentages) / len(gaze_percentages)) * 99 * -1
        print(f"Average Gaze Percentage: {final_output:.2f}")
        return final_output
    else:
        print("No gaze data available.")
        return jsonify({"error": "No gaze data available."})
    

def classify_emotion(predictions):
    happiness_percentage = predictions[0][0] * 100
    # sadness_percentage = (1 - predictions[0][0]) * 100
    return happiness_percentage


def check_happiness(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        print(f"TensorFlow version: {tf.__version__}")
        emotion_model = load_model('/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/imageclassifier.h5')
        emotion_model.summary()
        # Check if the camera is opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return 1

        total_happiness_percentage = 0
        total_sadness_percentage = 0
        frame_count = 0
        gaze_percentage = eye_tracking_method(video_path)
        happiness_percentage = 0  # Moved the initialization here
        sadness_percentage = 1
        # gaze_percentage = eye_tracking_method(video_path)
        
        while True:
            # Capture a single frame
            ret, frame = cap.read()

            # Break the loop if no more frames are available
            if not ret:
                break

            # Call your eye tracking function on the entire frame
            if gaze_percentage < 0 or gaze_percentage == 100:
                print(gaze_percentage)
                return 1

            # Check if the person is looking (gaze_percentage > threshold, adjust the threshold accordingly)
            if gaze_percentage > 0:
                # Check if the frame is too dark or empty
                if np.mean(frame) < 10:
                    happiness_percentage = 1
                    # sadness_percentage = 99
                    return 1
                else:
                    # Resize the frame to match the input size of your emotion classification model
                    resized_frame = cv2.resize(frame, (256, 256))

                    # Normalize pixel values
                    resized_frame = resized_frame.astype('float32') / 255.0

                    # Expand dimensions to create a batch (assuming the model expects a batch input)
                    input_image = np.expand_dims(resized_frame, axis=0)

                    try:
                        predictions = emotion_model.predict(input_image)
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")
                        return 1

                    # Interpret the predictions
                    happiness_percentage = classify_emotion(predictions)
                    print('Analyzing')
                    # print(f'Happiness Percentage: {happiness_percentage:.2f}%')
                    # print(f'Sadness Percentage: {sadness_percentage:.2f}%')

                    # Accumulate the happiness and sadness percentages
                    total_happiness_percentage += float(happiness_percentage)
                    # print(type(total_happiness_percentage))
                    # total_sadness_percentage += sadness_percentage
                    frame_count += 1

        # Release the video capture
        cap.release()

        if frame_count > 0:
            # Calculate the average happiness and sadness percentages
            average_happiness = total_happiness_percentage / frame_count
            # average_sadness = total_sadness_percentage / frame_count

            print(f'Average Happiness Percentage: {average_happiness:.2f}%')
            # print(f'Average Sadness Percentage: {average_sadness:.2f}%')
            print(gaze_percentage)
            if(gaze_percentage > 99): return 1  # Fix the format specifier here
            average_result = (average_happiness + float(gaze_percentage)) / 2
            if(average_result > 99): return 1
            # print(f'Average Result: {average_result:.2f}')
            # print(type(average_result))
            return average_result
        else:
            print("No frames available for processing.")
            return 1

    except Exception as e:
        print(f"Error in check_happiness: {str(e)}")
        return 1

'''
def calculate_clarity():
    try:
        audio_file = "/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/output.mp3"
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)

        print(transcript.text)
        if(str(transcript.text) == 'none' or str(transcript.text) == ""):
            result = check_grammar(' yesterday.dont what that this')
        else:
            result = check_grammar(str(transcript.text))

        # return result if result > 1 else 1
        if(int(result) < 15): return 15
        else: return int(result)

    except Exception as e:
        # Handle the exception here
        print(f"Error in calculate_clarity: {str(e)}")
        return None  # or any default value you prefer



def calculate_boldness():
    try:
        audio_file = "/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/output.mp3"

        # Load the audio file
        y, sr = librosa.load(audio_file)

        # Calculate the spectrogram
        spectrogram = np.abs(librosa.stft(y))

        # Calculate the spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram)

        # Calculate the mean of the spectral contrast
        mean_contrast = np.mean(spectral_contrast)
        # print(type(mean_contrast))

        return (mean_contrast) * 2.2

    except Exception as e:
        # Handle the exception here
        print(f"Error in calculate_boldness: {str(e)}")
        return None  # or any default value you prefer





@app.route('/upload_video_new', methods=['POST'])
def upload_frontend_video():
    try:
 
        video_data = request.files['videoData']
        upload_video(video_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'Video uploaded successfully', 'result': 0}), 200


@app.route('/get_result', methods=['GET'])
def getResult():
    try:
        video_path = '/Users/sagni/OneDrive/Desktop/AI_Interview/backend/server/uploaded_video.webm'
        eye_contact = check_happiness(video_path)
        eye_contact = round(eye_contact,2)
        if(eye_contact == 1) :
            with app.app_context():
                # if(eye_contact == 1):
                return jsonify({ "eye_contact": eye_contact , "boldness" : 1, "clarity" : 1, "confidence" : 1, "overall" : 1})
                # else:
                #     return jsonify({ "eye_contact": eye_contact , "boldness" : boldness, "clarity" : clarity, "confidence" : confidence, "overall" : overall})
        voice_extraction_main()
        # voice_quality_analysis = voice_output.voice_quality()
        clarity = round(float(calculate_clarity()),2)
        boldness = round(float(calculate_boldness()),2)
        # clarity = round(clarity,2)
        # boldness = round(boldness,2)
        confidence = round((eye_contact+clarity+boldness)/3,2)
        # confidence = round(confidence,2)
        overall = round((eye_contact+boldness+clarity+confidence)//4,2)
        # overall = round(overall,2)
        with app.app_context():
            # print("clarity :"+clarity + "confidence: "+confidence +" boldness : "+boldness +" eye_contact : "+eye_contact + "Overall : "+overall)
            # return jsonify({eye_contact})
            return jsonify({ "eye_contact": eye_contact , "boldness" : boldness, "clarity" : clarity, "confidence" : confidence, "overall" : overall})
    except Exception as e:
        print(f"Error in check: {str(e)}")
        return jsonify({"error": str(e)}), 500

    

if __name__ == '__main__':
    ngrok_tunnel = ngrok.connect(5001)
    print('Public URL:', ngrok_tunnel.public_url)
    app.run(host='localhost', port=5001)