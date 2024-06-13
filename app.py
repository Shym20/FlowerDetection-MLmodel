
import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
from os import listdir
from PIL import Image

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style/>',unsafe_allow_html=True)

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('my_model.keras')

def classify_imagess(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcomee = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcomee

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    if(flower_names[np.argmax(result)]=='daisy'):
      desc = "Daisies are charming, cheerful flowers that have captured the hearts of many with their simple yet captivating beauty. Belonging to the Asteraceae family, daisies are known for their distinctive appearance characterized by a central disc surrounded by delicate, often white, petals. However, daisies can also bloom in a variety of other colors, including shades of pink, yellow, and even red.One of the most iconic features of daisies is their bright yellow center, which is composed of numerous tiny disk florets tightly packed together. This central disk is surrounded by a ring of ray florets, which are the delicate petals that give the daisy its classic appearance. These petals are often arranged in a symmetrical, star-like pattern, creating a visually striking contrast against the center." 
    elif(flower_names[np.argmax(result)]=='tulip'):
        desc = "Tulips are captivating flowers with a graceful elegance that has enchanted admirers for centuries. Belonging to the genus Tulipa, these vibrant blooms are renowned for their distinctive cup-shaped petals and slender, upright stems. Tulips come in a breathtaking array of colors, ranging from bold reds and sunny yellows to delicate pinks, purples, and whites, offering a kaleidoscope of beauty in gardens, parks, and floral arrangements. Originating from Central Asia, tulips have a rich cultural history, symbolizing love, prosperity, and new beginnings. Their enchanting allure and graceful demeanor make them a perennial favorite among flower enthusiasts and gardeners worldwide."
    elif(flower_names[np.argmax(result)]=='sunflower'):
        desc = "Sunflowers, with their radiant blooms and towering stalks, exude a vibrant energy that symbolizes warmth, vitality, and optimism. Helianthus annuus, as they are scientifically known, are renowned for their large, daisy-like flower heads, which consist of a dense arrangement of golden-yellow petals surrounding a central disk filled with hundreds of tiny florets. These remarkable flowers possess a unique trait called heliotropism, where young blooms turn their faces to follow the sun throughout the day, a behavior that inspired their name. Sunflowers are not only visually stunning but also hold cultural significance across many societies, representing adoration, loyalty, and happiness. Their seeds, rich in nutrients and oils, are a popular snack and ingredient in various culinary delights, while their towering stems and bright blooms make them a favorite choice for gardens, landscapes, and floral arrangements, adding a touch of sunshine wherever they grow."
    elif(flower_names[np.argmax(result)]=='dandelion'):
        desc = "Dandelions, often regarded as humble weeds, possess a delicate beauty and resilience that belies their common classification. Taraxacum officinale, as they are scientifically named, are recognizable for their distinctive yellow flowers and feathery seed heads that disperse with a breath of wind. These hardy plants thrive in a variety of environments, from lush meadows to cracks in urban sidewalks, making them ubiquitous across the globe. Despite their reputation as nuisances in lawns and gardens, dandelions have a rich history of culinary, medicinal, and cultural significance. Their nutritious leaves are edible and can be enjoyed in salads or cooked as greens, while their roots and flowers have been utilized in traditional herbal remedies for centuries. Symbolically, dandelions are often associated with resilience, transformation, and the fleeting beauty of life, as their bright blooms give way to delicate seed heads that dance in the breeze, reminding us of the cycle of growth and renewal."
    elif(flower_names[np.argmax(result)]=='rose'):
        desc = "Roses are arguably the most iconic and beloved flowers in the world, renowned for their timeless beauty, enchanting fragrance, and rich symbolism. Belonging to the genus Rosa, roses are part of the Rosaceae family and encompass over a hundred species, each with its own unique characteristics and variations. From ancient mythology to modern romance, roses have captivated humanity for centuries, serving as a symbol of love, passion, and elegance.One of the most distinguishing features of roses is their exquisite, symmetrical blooms, which typically consist of layers of delicate petals radiating from a central core. Roses come in a stunning array of colors, including classic shades like red, pink, white, and yellow, as well as more exotic hues such as lavender, peach, and bi-color varieties. The diversity of colors and fragrances ensures that there is a rose to suit every taste and occasion."

    return desc

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_imagess(uploaded_file))
    st.write(classify_images(uploaded_file))
    
    
folder = "C:/Users/dell/Downloads/Flower_Recogn/Images/daisy"

            
    

    

