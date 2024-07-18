from tensorflow.keras.models import model_from_json
import tensorflow.keras.metrics as metrics
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Paths to model architecture and weights
model_architecture_path = r'F:\sathwik\E3SEM2\Mini\Hate_Speech\model_architecture.json'
model_weights_path = r'F:\sathwik\E3SEM2\Mini\Hate_Speech\my_keras_model.h5'

# Load the model architecture from JSON
with open(model_architecture_path, 'r') as json_file:
    json_config = json_file.read()
model = model_from_json(json_config)

# Compile the model (including custom metrics)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', metrics.SpecificityAtSensitivity(0.5), metrics.AUC()])

# Load the model weights
model.load_weights(model_weights_path)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI setup
st.title("Bone Fracture Image Classification")

# Navigation bar
nav_selection = st.sidebar.radio("Navigation", ["Model Details", "Author", "Model Prediction"])

if nav_selection == "Model Details":
    st.header("Model Details")
    st.subheader("Model Architecture")
    st.markdown("""
    This model uses a Convolutional Neural Network (CNN) architecture for binary classification of Bone Fracture images.
    
    **Layers**:
    - Conv2D: Applies convolutional filters to extract features from input images.
    - BatchNormalization: Normalizes layer activations to stabilize training.
    - MaxPooling2D: Downsamples feature maps, preserving dominant features.
    - Dropout: Regularizes the model by randomly dropping neurons during training.
    - Dense: Fully connected layers for classification based on extracted features.
    
    **Optimizer**: Adam
    **Loss Function**: Binary Crossentropy
    **Metrics**: Accuracy, Specificity at Sensitivity 0.5, AUC
    """)

    st.subheader("Special Features")
    st.markdown("""
    - **Feature Hierarchies**: CNN builds progressively complex features from low-level to high-level representations.
    - **Weight Sharing**: Parameters in convolutional layers are shared across space, reducing model size.
    - **Translation Invariance**: Achieved through pooling layers, allowing detection of features regardless of position.
    - **Adaptive Learning Rates**: Adam optimizer adjusts learning rates for each parameter individually.
    - **Regularization**: Dropout adds noise to prevent overfitting, improving model generalization.
    """)

elif nav_selection == "Author":
    st.header("Author")
    st.subheader("Yalla Venkata Suresh (Sathwik)")
    st.markdown("""
    **About Me**:
    - **Education**: Final-year Computer Science and Engineering student at RGUKT IIIT Srikakulam.
    - **Experience**: Python Programming Intern at Oasis Infobyte, Machine Learning Intern at Codsoft.
    - **Skills**: Python, TensorFlow, Keras, Streamlit, and more.
    - **Projects**: Developed several projects including Bone Fracture Detection, Lung Cancer Detection, and more.
    - **Contact**: [LinkedIn](https://www.linkedin.com/in/suresh-yalla), [Portfolio](https://sathwiky579.github.io/Portfolio/Portfolio.html)
    - **Interests**: Deep Learning, Computer Vision, Natural Language Processing.
    """)

elif nav_selection == "Model Prediction":
    st.header("Model Prediction")
    st.subheader("Upload an Image for Classification")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)

        # Determine prediction label and confidence
        confidence = prediction[0][0]
        if confidence >= 0.5:
            prediction_label = "Not Fractured"
        else:
            prediction_label = "Fractured"
            confidence = 1 - confidence  # Adjust confidence for "Malignant"

        # Display the prediction
        st.write(f"Prediction: **{prediction_label}** with confidence **{confidence:.2f}**")

