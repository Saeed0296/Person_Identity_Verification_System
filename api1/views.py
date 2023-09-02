from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import face_recognition
from django.http import JsonResponse
import dlib
import base64
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os


def home(request):
    return render(request, 'home.html')

def calculate_eye_distance(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Load the shape predictor model
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Convert the image to numpy array
    image_np = np.array(image)

    # Detect faces
    detector = dlib.get_frontal_face_detector()
    faces = detector(image_np)

    # Ensure only one face is detected
    if len(faces) != 1:
        return None
    else:
        landmarks = predictor(image_np, faces[0])
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        distance_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        average_interocular_distance_mm = 65
        conversion_factor = average_interocular_distance_mm / distance_pixels
        distance_mm = distance_pixels * conversion_factor
        return distance_mm


def faces_match_view(request):
    if request.method == 'POST':
        folder1 = request.FILES.getlist('folder1')
        folder2 = request.FILES.getlist('folder2')
        true_labels = []
        predicted_labels = []

        for image1, image2 in zip(folder1, folder2):
            try:
                image1_array = np.asarray(bytearray(image1.read()), dtype=np.uint8)
                image2_array = np.asarray(bytearray(image2.read()), dtype=np.uint8)
                image1_cv2 = cv2.imdecode(image1_array, -1)
                image2_cv2 = cv2.imdecode(image2_array, -1)
            except Exception as e:
                return JsonResponse({'result': f"Error reading images: {e}"})
            
            try:
                eye_distance1 = calculate_eye_distance(image1_cv2)
                eye_distance2 = calculate_eye_distance(image2_cv2)

                if eye_distance1 is None or eye_distance2 is None:
                    return JsonResponse({'result': "Error: Exactly one face should be detected."})
            
                xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(xml_path)
            
                gray_image1 = cv2.cvtColor(image1_cv2, cv2.COLOR_BGR2GRAY)
                faces1 = face_cascade.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces1) > 0:
                    largest_face1 = max(faces1, key=lambda f: f[2] * f[3])
                    x1, y1, w1, h1 = largest_face1
                    extracted_face1 = image1_cv2[y1:y1+h1, x1:x1+w1]
                    extracted_face1_resized = cv2.resize(extracted_face1, (100, 100))
                    extracted_face1_rgb = cv2.cvtColor(extracted_face1_resized, cv2.COLOR_BGR2RGB)

                gray_image2 = cv2.cvtColor(image2_cv2, cv2.COLOR_BGR2GRAY)
                faces2 = face_cascade.detectMultiScale(gray_image2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces2) > 0:
                    largest_face2 = max(faces2, key=lambda f: f[2] * f[3])
                    x2, y2, w2, h2 = largest_face2
                    extracted_face2 = image2_cv2[y2:y2+h2, x2:x2+w2]
                    extracted_face2_resized = cv2.resize(extracted_face2, (100, 100))
                    extracted_face2_rgb = cv2.cvtColor(extracted_face2_resized, cv2.COLOR_BGR2RGB)
                else:
                    return JsonResponse({'result': "No faces found in the second image."})
                
            except Exception as e:
                return JsonResponse({'result': f"Error processing images: {e}"})
            
            try:
                try:
                    face_encodings1 = face_recognition.face_encodings(extracted_face1_rgb)[0]
                    face_encodings2 = face_recognition.face_encodings(extracted_face2_rgb)[0]
                except Exception as e:
                    return JsonResponse({'result': " No face found try different images"})
                

                threshold = 0.6

                face_distance = np.linalg.norm(np.array(face_encodings1) - np.array(face_encodings2))

                #true_labels.append(1)

                image1_filename = os.path.basename(image1.name)
                image2_filename = os.path.basename(image2.name)
                
                # Determine if the filenames are the same or different
                are_filenames_same = image1_filename == image2_filename

                # Set the true label based on filename match
                true_label = 1 if are_filenames_same else 0
                true_labels.append(true_label)
                predicted_labels.append(1 if face_distance < threshold else 0)

                if face_distance < threshold:
                    result = "The faces are a match!"
                else:
                    result = "The faces are not a match."

                extracted_face1_base64 = base64.b64encode(cv2.imencode('.png', extracted_face1_resized)[1]).decode()
                extracted_face2_base64 = base64.b64encode(cv2.imencode('.png', extracted_face2_resized)[1]).decode()
            

            except Exception as e:    
                return JsonResponse({'result': f"Error comparing faces: {e}"})
                pass
        
            tp = sum(t == p == 1 for t, p in zip(true_labels, predicted_labels))
            fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, predicted_labels))
            tn = sum(t == p == 0 for t, p in zip(true_labels, predicted_labels))
            fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, predicted_labels))

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0


        try:   
            print(true_labels)
            print(predicted_labels)
            confusion = confusion_matrix(true_labels, predicted_labels)
            class_names = ['No Match', 'Match']
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            confusion_img = io.BytesIO()
            plt.savefig(confusion_img, format='png')
            plt.close()
                
            context = {
                # Include other context data
                'result': result,
                'eye_distance1': eye_distance1,
                'eye_distance2': eye_distance2,
                'extracted_face1_base64': extracted_face1_base64,
                'extracted_face2_base64': extracted_face2_base64,
                'confusion_matrix_img': base64.b64encode(confusion_img.getvalue()).decode(),
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,

            }
            return render(request, 'result_templates.html', context)
            
        except Exception as e:
            return JsonResponse({'result': f"Error creating confusion matrix: {e}"})
    
    return render(request, 'upload_templates.html')



#Fcae comaprision for 2 Images only without confusion matrix
def face_match_view(request):
    if request.method == 'POST':
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']

        # Read the uploaded images using OpenCV
        try:
            image1_array = np.asarray(bytearray(image1.read()), dtype=np.uint8)
            image2_array = np.asarray(bytearray(image2.read()), dtype=np.uint8)
            image1_cv2 = cv2.imdecode(image1_array, -1)
            image2_cv2 = cv2.imdecode(image2_array, -1)
        except Exception as e:
            return JsonResponse({'result': f"Error reading images: {e}"})
        
        try:
            eye_distance1 = calculate_eye_distance(image1_cv2)
            eye_distance2 = calculate_eye_distance(image2_cv2)

            if eye_distance1 is None or eye_distance2 is None:
                return JsonResponse({'result': "Error: Exactly one face should be detected."})
            
        except Exception as e:
            return JsonResponse({'result': f"Error calculating eye distance: {e}"})


        # Load the Haar Cascade classifier for face detection
        try:
            xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(xml_path)
        except Exception as e:
            return JsonResponse({'result': f"Error loading Haar Cascade XML file: {e}"})

        # Process the images
        try:
            gray_image1 = cv2.cvtColor(image1_cv2, cv2.COLOR_BGR2GRAY)
            faces1 = face_cascade.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces1) > 0:
                largest_face1 = max(faces1, key=lambda f: f[2] * f[3])
                x1, y1, w1, h1 = largest_face1
                extracted_face1 = image1_cv2[y1:y1+h1, x1:x1+w1]
                extracted_face1_resized = cv2.resize(extracted_face1, (100, 100))  # Resize the image
                extracted_face1_rgb = cv2.cvtColor(extracted_face1_resized, cv2.COLOR_BGR2RGB)
            else:
                return JsonResponse({'result': "No faces found in the first image."})

            gray_image2 = cv2.cvtColor(image2_cv2, cv2.COLOR_BGR2GRAY)
            faces2 = face_cascade.detectMultiScale(gray_image2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces2) > 0:
                largest_face2 = max(faces2, key=lambda f: f[2] * f[3])
                x2, y2, w2, h2 = largest_face2
                extracted_face2 = image2_cv2[y2:y2+h2, x2:x2+w2]
                extracted_face2_resized = cv2.resize(extracted_face2, (100, 100))  # Resize the image
                extracted_face2_rgb = cv2.cvtColor(extracted_face2_resized, cv2.COLOR_BGR2RGB)
            else:
                return JsonResponse({'result': "No faces found in the second image."})
        except Exception as e:
            return JsonResponse({'result': f"Error processing images: {e}"})

        # Convert the extracted face images to face encodings
        try:
            face_encodings1 = face_recognition.face_encodings(extracted_face1_rgb)[0]
            face_encodings2 = face_recognition.face_encodings(extracted_face2_rgb)[0]
        except Exception as e:
            return JsonResponse({'result': f"Error encoding faces: {e}"})

        # Compare the face encodings
        try:
            face_distance = np.linalg.norm(np.array(face_encodings1) - np.array(face_encodings2))

            # Threshold for determining if the faces are the same person
            threshold = 0.6
            if face_distance < threshold:
                result = "The faces are a match!"
            else:
                result = "The faces are not a match."
                
            extracted_face1_base64 = base64.b64encode(cv2.imencode('.png', extracted_face1_resized)[1]).decode()
            extracted_face2_base64 = base64.b64encode(cv2.imencode('.png', extracted_face2_resized)[1]).decode()
            #context = {'result': result}
            context = {
                'result': result,
                'extracted_face1_base64': extracted_face1_base64,
                'extracted_face2_base64': extracted_face2_base64,
                'eye_distance1': eye_distance1,
                'eye_distance2': eye_distance2,
            }
            return render(request, 'result_template.html', context)
        except Exception as e:
            return JsonResponse({'result': f"Error comparing faces: {e}"})
    
    return render(request, 'upload_template.html')
