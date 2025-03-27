# Face Recognition System - README

## Prerequisites
Before running the Face Recognition System, ensure that you have the following prerequisites installed:

### **1. Python (>= 3.7)**
Ensure Python is installed. You can download it from [Python's official website](https://www.python.org/downloads/).

```bash
python --version
```

### **2. Install Required Python Libraries**
Run the following command to install the necessary dependencies:

```bash
pip install opencv-python face-recognition numpy flask watchdog
```

### **3. Install `dlib` (if required)**
`face-recognition` depends on `dlib`. If you encounter errors, install `dlib` manually:

```bash
pip install dlib
```

### **4. Ensure Camera Permissions**
For real-time face recognition, the application requires access to your system's camera. Ensure the camera is connected and accessible.

### **5. Create Necessary Directories**
Make sure the `database` folder exists to store known face images:

```bash
mkdir database
```

## Running the Application
1. Start the Flask server by running:

```bash
python FaceRecog.py
```

2. The application will open automatically in your web browser at:
   ```
   http://127.0.0.1:3000/
   ```

## Features
- Upload images to the `database` folder via the web interface.
- Automatically detects new faces in the database folder.
- Real-time face recognition using OpenCV and Flask.
- Web-based interface for easy interaction.

## Notes
- Ensure images added to the `database` folder are clear, front-facing images for accurate recognition.
- If encountering issues, restart the application and ensure all dependencies are correctly installed.

## Troubleshooting
- **Camera not detected?** Ensure it's properly connected and accessible.
- **Face not recognized?** Check if the image in the `database` is clear and well-lit.
- **Module errors?** Try reinstalling dependencies using:
  ```bash
  pip install --upgrade --force-reinstall -r requirements.txt
  ```

