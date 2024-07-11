# **Automated Filling Line with Pick and Place Robot**
### **Project Overview**
    This project aims to develop an automated filling line integrated with a pick and place robot. The primary focus is on the integration of a computer vision model capable of detecting bottles, classifying their colors, and checking whether the bottle's water level, label, and cap are correct. The project employs anomaly detection techniques to ensure quality control.

### **Features**
- Bottle Detection: Identifies the presence of bottles on the filling line.
- Color Classification: Classifies the color of each bottle.
- Water Level Detection: Checks if the water level in the bottle is correct.
- Label and Cap Inspection: Verifies the integrity of the bottle's label and cap using anomaly detection.

### **Technologies Used**
- PyTorch: For building and training the deep learning models.
- Ultralytics YOLO v5: For object detection and classification.
- OpenCV: For computer vision tasks and image processing.
- Supervision: For enhancing the performance and accuracy of the vision model.

### **Workflow**
- Data Collection: Gathered a comprehensive dataset of bottles with varying colors, water levels, labels, and caps.
- Data Annotation: Annotated the collected data to mark regions of interest and classify different attributes.
- Data Preprocessing: Preprocessed the annotated data to make it suitable for training the model.
- Model Training: Trained the computer vision model using PyTorch and YOLO v5.
- Model Deployment: Deployed the trained model to integrate it with the automated filling line system.

### **Installation**

- Clone the repository:
    - `git clone https://github.com/OmarGx100/automated-filling-line.git`

- Navigate to the project directory:
    - cd automated-filling-line
    - Install the required dependencies:
        - `pip install -r requirements.txt`

- Usage
    - Prepare the environment and ensure all dependencies are installed.
    - Run the model for bottle detection and classification:
        - `python App.py`



## **Contributing**
We welcome contributions to improve this project. Please fork the repository and create a pull request with your changes.

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any questions or inquiries, please contact [OmarAbdelRahman] at [omarmahmoudgx100@gmail.com].
