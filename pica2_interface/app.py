import streamlit as st
import requests
import cv2
import numpy as np
import sys


st.set_page_config(layout="centered", page_title="PICA-2 AI ART")

st.title("PICA-2")


st.write(
    "Abstract AI Art via Computational Creativity")

left,right = st.columns(2)





left.write("Fill in the data:")
form = left.form("template_form")



color = form.multiselect(
        'Select a Style',
        ['prism_r','hot','Paired','Set1', 'flag','gist_ncar','seismic','gray_r', 'coolwarm','viridis'],max_selections = 1)

image1 = form.multiselect(
        'Select your Categories',
        ['apple', 'mountain', 'cloud', 'butterfly', 'house', 'door'],max_selections = 2 )

name = form.text_input('What is your name?')



#query = {"color": , "alpha": , "beta": }
right.write("Here is your generated image:")
if form.form_submit_button("Generate Image"):
    if len(name) > 8:
        form.error("You can't use more than 8 characters")
        sys.exit()
    else:
        alpha = image1[0]
        beta = image1[1]
        color = color[0]
        name = name

    data = {"alpha": alpha, "beta": beta, "color": color, "name":name}
    url = f"http://127.0.0.1:8000/super?alpha={alpha}&beta={beta}&color={color}&name={name}&noise_dim=100&num_examples=1"
    #right.write("Heres your generated image:")

    response = requests.post(url)


    #image = Image.open(io.BytesIO(response.content)) #Nicole's original
    #right.image(image, width = 300) #Nicole's original

    #st.image(Image.open("/Users/ds_janf/code/janmfriedli/PICA-2/PICA-2/api/one.png"))
    output = response.content #NEW
    #st.markdown(response.status_code)
    if response.status_code == 200:
        img = np.frombuffer(output , np.uint8)
        img = cv2.imdecode(img , cv2.IMREAD_UNCHANGED)
        right.image(img, width = 300)
    elif response.status_code == 429:
        st.warning("Hey slow down there!")
    else:
        st.warning("Something went terribly wrong.")
        st.warning("Please wait before trying again.")

    #plt.imshow(im) #NEW
    #plt.axis('off') #NEW



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXMCYn5iGIL9UBihQb5oNUx0JAAZkyD4E5kKcUuJkYPBpu9PVPjPu6s0Ddw863NcV6UZo&usqp=CAU");
             background-attachment: fixed;
             background-size: cover
         }}

         [data-testid="stHeader"] {{
            background-color: #00d4ff00
         }}

         </style>
         """,
         unsafe_allow_html=True
     ) # background (to be changed)

#add_bg_from_url()
