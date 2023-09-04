# PIV-Personal-Identity-Verification-System
# Face Recognition and Comparison Django App

This Django app is designed to perform face recognition and comparison using various libraries like OpenCV, face_recognition, dlib, and more. It offers three main functionalities: Face Matching for Two Images, and Face Matching of two folders containing  person images and CNICs.

## Installation

1. Clone the repository:https://github.com/romailafzal/PIV-Personal-Identity-Verification-

2. Create a virtual environment (recommended)

3. Install the project dependencies:   pip install -r requirements.txt

4. Run the Django development server:  python manage.py runserver


## Usage

1. Access the app through your web browser by visiting `http://localhost:8000`.

2. **Home Page:** Landing page with buttons having links to other functionalities.

3. **Face Matching for Two Images:**
  - Upload two images.
  - Compare the faces and view the result.

4. **About Confusion Matrix**
  - Face Matching of two folders containing person images and CNICs:
  - Match faces, calculate metrics, and view the confusion matrix.
    In the `faces_match_view` function, the code processes images from two different folders and labels them as true or false based on whether the images have the same filenames. If images in both folders have the same filename, they are labeled as true, and if the filenames are different, they are labeled as false. It's important to note that this filename-based labeling doesn't guarantee the images are actually similar; the similarity can only be inferred from the filenames. Users should ensure that images with matching filenames do indeed contain similar faces.

5. **Eye Distance Calculation**
  The current version of the code calculates eye distance using pixels, which might not yield the most accurate results. We acknowledge that this approach has limitations and plan to update it as we achieve better results through more sophisticated techniques. Inaccuracies may arise due to variations in image resolutions and facial poses. Stay tuned for improvements in this area.

## Contributing

  Contributions are welcome! If you have any ideas or improvements, please open an issue or submit a pull request.

## Credits

  This project was developed by Team Alpha. It utilizes various open-source libraries and technologies.
