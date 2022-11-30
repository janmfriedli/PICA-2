import streamlit as st

st.set_page_config(layout="centered", page_icon="🎨", page_title="PICA-2 AI ART")

st.title("🎨 PICA-2")


st.write(
    "Abstract AI Art via Computational Creativity")

left,right = st.columns(2)

left.write("Fill in the data:")
form = left.form("template_form")
color = form.multiselect(
        'Select a Color',
        ['Green', 'Yellow', 'Red', 'Blue'],)
image1 = form.multiselect(
        'Select your Categories',
        ['Apple', 'Eiffel Tower', 'Fish', 'Squirrel', 'House'],)


generate = form.form_submit_button("Generate Image")

right.write("Here's your generated image!:")
right.image("pica2 logo.png", width=300) # image will be here with api call


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXMCYn5iGIL9UBihQb5oNUx0JAAZkyD4E5kKcUuJkYPBpu9PVPjPu6s0Ddw863NcV6UZo&usqp=CAU");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     ) # background (to be changed)

add_bg_from_url()
