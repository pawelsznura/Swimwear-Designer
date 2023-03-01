import streamlit as st
from PIL import Image
from io import BytesIO
import os
import base64

st.set_page_config(layout="wide", page_title="Image Galery")

st.write("## Image Gallery")

# get number of generated img 
# Get a list of all files in the folder
files = os.listdir("created_images")

# Get the number of files in the folder
num_files = len(files)

st.write('The select a number between 1 and ', num_files)
selected_number = st.number_input('', value=num_files, min_value=1, max_value=num_files, step=1, key="number_input")




col1, col2 = st.columns(2)

with col1:
    st.image("created_images/"+str(selected_number)+".png")


with col2:
    f = open("created_images_text/"+str(selected_number)+".txt", "r")
    text = f.read()

    lines = text.split("\n")
    # Loop over the lines and limit the number of characters in the line starting with "prompt:"
    for i, line in enumerate(lines):
        if line.startswith("prompt:"):
            max_chars_per_line = 57
            prompt_display = "\n".join([line[i:i+max_chars_per_line] for i in range(0, len(line), max_chars_per_line)])
            lines[i] = prompt_display

    # Join the lines with line breaks and display the text
    text_display = "\n".join(lines)

    st.text(text_display)
    st.write("inspiration image")

    insp_img_path = text.split("input image: ")[1].split()[0]

    if insp_img_path == "output":
        st.write("no image to show")
    else:
        st.image(insp_img_path, width=250)



