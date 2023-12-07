from PIL import Image
import streamlit as st
from models import IR50_EVR_AgeRM_GP
import face_recognition
import torch
from utils import get_data_specs, download_chpts, predict
from pillow_heif import register_heif_opener

register_heif_opener()


@st.cache_resource
def select_model(data):
    age_num, age_labels = get_data_specs(data)
    
    model = IR50_EVR_AgeRM_GP(age_num=age_num)
    chpt_path = download_chpts(data)
    model.load_state_dict(torch.load(chpt_path, map_location=torch.device("cpu")))
    
    return model, age_labels


@st.cache_resource
def detect_faces(file):
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    return image, face_locations


### PAGE SETUP ###
st.set_page_config(page_title="Facial Age and Gender Prediction",
                   page_icon=":bust_in_silhouette:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

st.markdown("<h1 style='text-align: center;'>Face-based Age and Gender Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>👤 → 🤖 → 💬</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center;'>Upload a photo of yourself to see how the computer interprets your facial features!</h4>",
    unsafe_allow_html=True)
st.markdown('___')


### SIDEBAR ###
with st.sidebar:
    st.header('About')
    st.write(
        "This is an app that performs `face-based` `age estimation` and `gender prediction` using deep learning "
        "model proposed in our paper. You can check the code for model implementation "
        "[here](https://github.com/SeoulTech-HCIRLab/Relative-Age-Position-Learning.git).")
    st.markdown('___')
    st.write("This streamlit app is created by [sevahun](https://github.com/sevahun). "
             "You can check out the app source code [here](https://github.com/sevahun/age-gender-prediction-app).")

    """
        [![Stars](https://img.shields.io/github/stars/sevahun/age-gender-prediction-app.svg?logo=github&style=social)](https://gitHub.com/sevahun/age-gender-prediction-app)
    """

    st.markdown('___')
    st.header('External Sources')
    st.write('Detection and alignment of faces are done with the help of  '
             '[face recognition](https://pypi.org/project/face-recognition/) library.')


### BODY ###
st.markdown("<h4 style='text-align: left;'>Complete the following steps to perform a prediction:</h4>",
            unsafe_allow_html=True)

exp_1, exp_2, exp_3 = (st.expander("⚙️ Adjust model settings"),
                       st.expander("📄 Upload an image"),
                       st.expander("👤 Make a prediction"))

with exp_1:
    st.write('Select model checkpoints or read about them in details by switching between tabs!')
    setting_tab, details_tab = st.tabs(["Setting", "Details"])

    with setting_tab:
        st.write('Select model checkpoints:')
        data = st.radio("checkpoints", options=["AgeDB", "AFAD"], label_visibility="collapsed")
        model, age_labels = select_model(data.lower())

        if model:
            st.write('✅ Checkpoints are loaded.  \n 👇 You can move to the next step!')

    with details_tab:
        st.write('Switch between tabs to read more about model checkpoints!')
        agedb_tab, afad_tab = st.tabs(["AgeDB", "AFAD"])
        with agedb_tab:
            st.write('The AgeDB is a dataset that contains facial images of various '
                     'famous people captured under `real-world conditions` (i.e., having different poses, bearing '
                     'various expressions, containing noise and occlusions, etc.). The dataset includes images for '
                     'ages from `1` to `101` years.')
            st.write('❗️If you want to test the model on `in-the-wild` images that capture people with ages that '
                     'entail a `full lifespan` (children, adults, seniors), use this checkpoint.')

        with afad_tab:
            st.write('The AFAD is a dataset that contains facial images (mostly selfie photos) of `asian '
                     'people` from `15` to `72` years.')
            st.write('❗️If you want to test the model on images that capture people of `asian ethnicity`, use this '
                     'checkpoint. Note that a prediction of this model checkpoint is unreliable for ages '
                     '`under 15` and `above 72` years.')

with exp_2:
    st.write('👇 Upload an image you want to do a prediction on!')
    file = st.file_uploader("image load", type=["jpg","png", "heic"], label_visibility="collapsed")
    st.write('❗️ For user who accessed through mobile device:  \n Please note that the mobile Safari browser automatically converts the `HEIC` image to `JPEG` image and causes the rotation of the image. The wrong image orientation leads to a misdetection of faces!')

with exp_3:
    st.write('Check the detected faces or make sure if the original image is uploaded correctly by switching between tabs!')
    faces_tab, original_tab = st.tabs(["Faces", "Uploaded Image"])

    if file is None:
        faces_tab.write(f"No image is uploaded")
        original_tab.write(f"No image is uploaded")
    else:
        exp_2.write('✅ Image is uploaded.  \n 👇 You can move to the next step!')
        image, face_locations = detect_faces(file)
        faces_tab.write(f"Face detection is completed.  \n Number of detected faces: `{len(face_locations)}`.")
        original_tab.write('👆 Move to the Faces tab to make predictions on faces detected on this image.')
        original_tab.image(file, width=200, caption="Uploaded image.")

        if not face_locations:
            faces_tab.write("Please upload another image.")
        else:
            with faces_tab:
                st.write(f"👇 Good! Now, you can choose the face you want to do a prediction on!")
                indices = [i + 1 for i in range(0, len(face_locations))]
                faces = []

                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    face_image = image[top:bottom, left:right]
                    face_image = Image.fromarray(face_image)
                    faces.append(face_image)

                st.image(faces, caption=indices, width=60)
                st.write('Select a face image:')
                face = st.radio("face image", indices, label_visibility="collapsed")
                face = faces[face - 1]
                clicked = st.button(f"Predict")

                if clicked:
                    with st.spinner(text='In progress'):
                        age, gender, gender_label = predict(face, model, age_labels, data.lower())
                    st.write('✅ Prediction is completed.')
                    st.write('Prediction results:')
                    col_1, col_2 = st.columns(3)[0].columns(2)
                    col_1.metric(label="Age", value=f"{round(age.item(), 2)}", delta="Years")
                    col_2.metric(label="Gender", value=f"{gender}", delta=f"{gender_label}")


### END ###
st.markdown('___')
st.write("Thanks for going through this app! If you are interested in how this app was developed, more info is "
         "[here](https://github.com/sevahun/age-gender-prediction-app).  "
         "\n Any errors or suggestions? "
         "[Let me know by opening a GitHub issue](https://github.com/SeoulTech-HCIRLab/Relative-Age-Position-Learning/issues)!  \n"
         "If you liked this app, please leave a ⭐!")
