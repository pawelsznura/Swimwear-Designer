import streamlit as st
from PIL import Image
from io import BytesIO
import os
import base64
import pandas as pd


# functions

def display_img_data(selected_number):
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

    rating = img_rat.loc[selected_number-1, "score"]
    st.write(f"Rating: {rating}")

    st.write("inspiration image")

    insp_img_path = text.split("input image: ")[1].split()[0]

    if insp_img_path == "output":
        st.write("no image to show")
    else:
        st.image(insp_img_path, width=250)




# MAIN

st.set_page_config(layout="wide", page_title="Image Galery")

st.write("## Image Gallery")

# get number of generated img 
# Get a list of all files in the folder
files = os.listdir("created_images")

# Get the number of files in the folder
num_files = len(files)

# get image rating data
# columns: image, score
img_rat = pd.read_csv("img_evaluation.csv")

# print(img_rat["score"].to_string())



# select img  rating 
rating_option = st.selectbox(
    'Select Image quality',
    ('All', '5', '4', '3', '2', '1', '0'))

if rating_option == 'All':
    selected_number = st.number_input('Select an image number between 1 and ' + str(num_files), value=num_files, min_value=1, max_value=num_files, step=1, key="number_input")
else:
    # Filter the img_rat DataFrame based on the selected rating
    img_rat_filtered = img_rat[img_rat["score"] == int(rating_option)]

    if img_rat_filtered.empty:
        st.write("No images with rating " + rating_option)
    else:
        # Get the list of image numbers with the selected rating
        img_numbers = img_rat_filtered["image"].tolist()

        # Display the list of image numbers with the selected rating
        selected_number = st.selectbox('Select an image number', img_numbers)

col1, col2 = st.columns(2)

with col1:
    st.image("created_images/"+str(selected_number)+".png")


with col2:
    display_img_data(selected_number)

    