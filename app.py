import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')  # Ganti dengan path model Anda

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses the image to be ready for model prediction.
    """
    try:
        image = image.convert("RGB")  # Convert to RGB
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.float32)  # Convert to array
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🖼️", layout="centered")
    st.title("🖼️ Klasifikasi Gambar dengan MobileNetV2")
    st.markdown(
        """
        **Model Klasifikasi Spesies Ikan**
        
        Aplikasi ini menggunakan data spesies ikan dengan 3 label:
        - **Label 1:** Chromis Chrysura
        - **Label 2:** Dascyllus Reticulatus
        - **Label 3:** Plectroglyphidodon Dickii
        """
    )

    uploaded_file = st.file_uploader("Upload gambar ikan (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            st.warning("⏳ Memproses gambar...")

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Predict using the model
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            probability = prediction[0][class_index] * 100  # Convert to percentage

            # Define labels
            labels = [
                "Label 1 (Chromis Chrysura)", 
                "Label 2 (Dascyllus Reticulatus)", 
                "Label 3 (Plectroglyphidodon Dickii)"
            ]
            predicted_label = labels[class_index]

            # Display prediction
            st.success(f"🎉 **Prediksi:** {predicted_label}")
            st.info(f"**Probabilitas:** {probability:.2f}%")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
