import cv2
from PIL import Image
import face_recognition as fr
import streamlit as st

st.set_page_config(layout="wide", page_title="face recogniser")
st.title("Face recognition System")
st.subheader("recognise face and compare them")

st.markdown("upload two images which you want to match!")

def load_image(image_file):
    img = Image.open(image_file)
    return img


def display(res):
    if res[0] == True:
        st.subheader("face matched 🎉🎉")
        st.text("congrats!!")
    else:
        st.subheader("face not matched")
        st.text("sorry!!")


image_1 = st.file_uploader("upload first image", type=["jpg"])
image_2 = st.file_uploader("upload second image", type=["jpg"])

if image_1 and image_2 is not None:
    # image1 = load_image(image_1)
    # image2 = load_image(image_2)

    #saving image file

    with open("image1.jpg", "wb") as f:
        f.write(image_1.getbuffer())

    with open("image2.jpg", "wb") as f:
        f.write(image_2.getbuffer())

    imageone = fr.load_image_file('image1.jpg')
    imagetwo = fr.load_image_file('image2.jpg')

    # converting color of images from bgr to rgb
    imageone = cv2.cvtColor(imageone, cv2.COLOR_BGR2RGB)
    imagetwo = cv2.cvtColor(imagetwo, cv2.COLOR_BGR2RGB)
    try:
        face1 = fr.face_locations(imageone)[0]
        face2 = fr.face_locations(imagetwo)[0]


        cv2.rectangle(imageone, (face1[3], face1[0]), (face1[1], face1[2]), (255, 0, 0), 4)
        cv2.rectangle(imagetwo, (face2[3], face2[0]), (face2[1], face2[2]), (255, 0, 0), 4)

        # encoding image one

        imageone_encode = fr.face_encodings(imageone)[0]
        # encoding image two
        imagetwo_encode = fr.face_encodings(imagetwo)[0]


        res = fr.compare_faces([imageone_encode], imagetwo_encode)
        print (res)
        display(res)
        st.text(" ")
        st.text(" ")
        imageone = cv2.cvtColor(imageone, cv2.COLOR_BGR2RGB)
        imagetwo = cv2.cvtColor(imagetwo, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.image(imageone, width=450)
        with col2:
            st.image(imagetwo, width=450)

    except IndexError as e:
        st.subheader("sorry not able to detect face in the image !!")

