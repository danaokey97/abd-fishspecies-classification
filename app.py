import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')  # Ganti dengan path model Anda jika berbeda

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")  # Convert image to RGB
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Main Streamlit app
def main():
    st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🖼", layout="centered")
    st.title("🖼 Klasifikasi Gambar dengan MobileNetV2")
    st.markdown(
        """
        Pada Klasifikasi ini menggunakan data spesies ikan yang memiliki 3 label : 
        - Label 1 ( Chromis Chrysura ) 
        - Label 2 ( Dascyllus Reticulatus )
        - Label 3 ( Plectroglyphidodon Dickii )
        """)
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(135deg, #f3f4f6, #e2e8f0);
            padding: 2rem;
            border-radius: 10px;
        }
        .uploadedImage {
            border: 2px solid #1f77b4;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .predictionLabel {
            font-size: 1.2rem;
            color: #1f77b4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Image :", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.markdown("<div class='uploadedImage'>", unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.warning("\nMemproses gambar...")
        processed_image = preprocess_image(image)

        # Prediksi gambar
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)

        # Anda dapat mengganti label ini dengan label yang sesuai dengan model Anda
        labels = ['Label 1 ( Chromis_Chrysura )', 'Label 2 ( Dascyllus_Reticulatus )', 'Label 3 ( Plectroglyphidodon_Dickii )']  # Sesuaikan dengan label spesifik model Anda
        predicted_label = labels[class_index]

        st.write("<hr>", unsafe_allow_html=True)
        st.markdown(f" <h3> Prediksi : </h3> <p> <strong class='predictionLabel'>{predicted_label}</strong></p>", unsafe_allow_html=True)
        st.markdown(f" <h3> Probabilitas : </h3> <p> <strong class='predictionLabel'>{prediction[0][class_index]:.2f} % </strong></p>", unsafe_allow_html=True)
        st.write("<hr>", unsafe_allow_html=True)

        st.success("Berhasil dekk!!")

if __name__ == "__main__":
    main()


# # Main Streamlit app
# def main():
#     st.title("Klasifikasi Gambar")
#     st.write("Unggah gambar untuk melihat tampilannya.")

#     uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Tampilkan gambar yang diunggah
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
#         st.write("\nGambar berhasil diunggah. Model klasifikasi belum tersedia.")

# if _name_ == "_main_":
#     main()
