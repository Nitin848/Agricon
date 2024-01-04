"""import pickle
import numpy as np
import sklearn
import requests, json

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

api_key = "9d7cde1f6d07ec55650544be1631307e"
city="Raichur"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
# city_name = input("Enter city name : ")
complete_url = base_url + "appid=" + api_key + "&q=" + city
response = requests.get(complete_url)
x = response.json()
if x["cod"] != "404":
    y = x["main"]
    current_temperature = y["temp"] - 273.15
    current_pressure = y["pressure"]
    current_humidity = y["humidity"]
    z = x["weather"]
    weather_description = z[0]["description"]

    print(" Temperature (in kelvin unit) = " +
          str(current_temperature) +
          "\n atmospheric pressure (in hPa unit) = " +
          str(current_pressure) +
          "\n humidity (in percentage) = " +
          str(current_humidity) +
          "\n description = " +
          str(weather_description))

else:
    print(" City Not Found ")
data = np.array([[100, 200, 50, current_temperature, current_humidity, 2, 150]])
my_prediction = crop_recommendation_model.predict(data)
final_prediction = my_prediction[0]
print(final_prediction)"""

import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model

class_names21 = pd.read_csv(
    "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/21_classnames.csv")
class_names_21 = list(class_names21["21_classnames"])
class_names2 = ["Healthy", "Unhealthy"]
diseases_preventation = pd.read_csv(
    "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/diseases preventation.csv")


def load_image(image_file):
    img = Image.open(image_file)
    return img


# Preprocess for healthy and unhealthy and 21 plant
def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    return pred_class


model = tf.keras.models.load_model(
    "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/plant_status_model1.h5")
model1 = tf.keras.models.load_model(
    "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/Plant_Leaf_identification_model1.h5")


def diseases_prediction(file_name, model, class_names):
    image = tf.keras.preprocessing.image.load_img(file_name)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    # Get the predicted class
    if len(predictions[0]) > 1:  # check for multi-class
        pred_class = class_names[predictions.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(predictions)[0][0])]  # if only one output, round
    return pred_class


def resize_image(file_name):
    image = cv2.imread(file_name)
    img = cv2.resize(image, (224, 224))
    cv2.imwrite(file_name, img)


def prediction(plant_name, file_name):
    resize_image(file_name)
    if plant_name == "Apple":
        Apple_model = tf.keras.models.load_model(
            "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/Apple_model1.h5")
        Apple_diseases = ['Apple_scab', 'Black_rot', 'Cedar_apple_rust']
        diseases = diseases_prediction(file_name, Apple_model, Apple_diseases)
        return diseases

    elif plant_name == "Grape":
        Grape_model = tf.keras.models.load_model(
            "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/Grapes_model1.h5")
        Grape_diseases = ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)']
        diseases = diseases_prediction(file_name, Grape_model, Grape_diseases)
        return diseases
    elif plant_name == "Tomato":
        Tomato_model = tf.keras.models.load_model(
            "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/tomato_model1.h5")
        Tomato_diseases = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
                           'Spider_mites_Two_spotted_spider_mite', 'Target_Spot', 'YellowLeaf__Curl_Virus',
                           'mosaic_virus']
        diseases = diseases_prediction(file_name, Tomato_model, Tomato_diseases)
        return diseases
    elif plant_name == "Corn":
        Corn_model = tf.keras.models.load_model(
            "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/corn_model1.h5")
        Corn_diseases = ['Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight']
        diseases = diseases_prediction(file_name, Corn_model, Corn_diseases)
        return diseases
    elif plant_name == "Potato":
        Potato_model = tf.keras.models.load_model(
            "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/Models/potato_model2.h5")
        Potato_diseases = ['Potato___Early_blight', 'Potato___Late_blight']
        diseases = diseases_prediction(file_name, Potato_model, Potato_diseases)
        return diseases
    else:
        diseases = "Diseases are not trained for this Plant"
        st.write(diseases)


st.title("Go TechFarMiss")
st.subheader("Plant Disease Detection")

st.markdown("Upload an image of the plant leaf")

# Uploading the dog image

image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
submit = st.button('Predict')
if image_file is not None:
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    img = load_image(image_file)
    st.image(img, height=250, width=250)
    with open(
            os.path.join("C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/temp/",
                         image_file.name), "wb") as f:
        f.write(image_file.getbuffer())
    path = "C:/Users/rvcr7/Downloads/Plant_Disease_Flask_App/Plant_Disease_Flask_App/Plant_status/temp/" + str(
        image_file.name)
    plant_name = pred_and_plot(model1, path, class_names_21)
    st.write("**Plant_name**:  " + str(plant_name))
    plant_status = pred_and_plot(model, path, class_names2)
    st.write("**Plant_status**:  " + str(plant_status))
    if plant_status == "Healthy":
        st.write("There is no need to spray pesticides on the leaf because it is healthy.")
    else:
        diseases = prediction(plant_name, path)
        if diseases == None:
            pass
        else:
            st.write("**Diseases**:  " + str(diseases))
            st.write("**Precautions:**")
            index_value = diseases_preventation.Diseases_Name[
                diseases_preventation.Diseases_Name == str(diseases)].index.tolist()
            st.write(diseases_preventation.iloc[index_value[0]]["Precautions"])




