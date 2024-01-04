import numpy as np
#import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from werkzeug.utils import redirect

#from utils.disease import disease_dic
#from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io



app = Flask(__name__)


fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }



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

def resize_image(file_name):
    image = cv2.imread(file_name)
    img = cv2.resize(image, (224, 224))
    cv2.imwrite(file_name, img)

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



def prediction(plant_name, file_name):
    resize_image(file_name)
    if plant_name == "Apple":
        Apple_model = tf.keras.models.load_model("Data/Apple_model1.h5")
        Apple_diseases = ['Apple_scab', 'Black_rot', 'Cedar_apple_rust']
        diseases = diseases_prediction(file_name, Apple_model, Apple_diseases)
        return diseases

    elif plant_name == "Grape":
        Grape_model = tf.keras.models.load_model("Data/Grapes_model1.h5")
        Grape_diseases = ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)']
        diseases = diseases_prediction(file_name, Grape_model, Grape_diseases)
        return diseases
    elif plant_name == "Tomato":
        Tomato_model = tf.keras.models.load_model("Data/tomato_model1.h5")
        Tomato_diseases = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
                           'Spider_mites_Two_spotted_spider_mite', 'Target_Spot', 'YellowLeaf__Curl_Virus',
                           'mosaic_virus']
        diseases = diseases_prediction(file_name, Tomato_model, Tomato_diseases)
        return diseases
    elif plant_name == "Corn":
        Corn_model = tf.keras.models.load_model("Data/corn_model1.h5")
        Corn_diseases = ['Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight']
        diseases = diseases_prediction(file_name, Corn_model, Corn_diseases)
        return diseases
    elif plant_name == "Potato":
        Potato_model = tf.keras.models.load_model("Data/potato_model2.h5")
        Potato_diseases = ['Potato___Early_blight', 'Potato___Late_blight']
        diseases = diseases_prediction(file_name, Potato_model, Potato_diseases)
        return diseases
    else:
       diseases = "Diseases are not trained for this Plant"
#      return "Diseases are not trained for this Plant"


class_names21 = pd.read_csv("Data/21_classnames.csv")
class_names_21 = list(class_names21["21_classnames"])
class_names2 = ["Healthy", "Unhealthy"]
diseases_preventation = pd.read_csv("Data/diseases preventation.csv")

model = tf.keras.models.load_model("Data/plant_status_model1.h5")
model1 = tf.keras.models.load_model("Data/Plant_Leaf_identification_model1.h5")
path='C:/Users/sunil kumar/OneDrive/Desktop/New folder (5)/disease detection/plant.png'







crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_recommend')
def crop_recommend():
    return render_template('crop.html')

@app.route('/crop_predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city_name = request.form.get("city")


        api_key = "9d7cde1f6d07ec55650544be1631307e"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()
        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"] - 273.15
            current_pressure = y["pressure"]
            current_humidity = y["humidity"]
            z = x["weather"]
            weather_description = z[0]["description"]
        #current_temperature=24
        #current_humidity=37
        data = np.array([[N, P, K, current_temperature, current_humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

    return render_template('crop_result.html',prediction_text=final_prediction)

@app.route('/fertilizer_recommend')
def fertilizer_recommend():
    return render_template('fertilizer.html')

@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_prediction():
    if request.method=='POST':
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        df = pd.read_csv('Data/fertilizer.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"
        response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer_result.html',prediction_result=response)

@app.route('/disease_prediction')
def disease_prediction():
    return render_template('disease.html')

@app.route('/disease_prediction',methods=['GET','POST'])
def disease_prediction_result():
    if request.method == 'POST':
        file = request.files['file1']
        image_file = file.filename
        file_path = os.path.join('C:/Users/sunil kumar/OneDrive/Desktop/New folder (5)/disease detection/temp/',image_file)
        #file.save(file_path)
        plant_name = pred_and_plot(model1, file_path, class_names_21)
        plant_status = pred_and_plot(model, file_path, class_names2)

        if plant_status == "Healthy":
            info = "There is no need to spray pesticides on the leaf because it is healthy."
            return render_template('disease_result.html',cond=0, plant=plant_name, status=plant_status, diseases_name=info)
        else:
            diseases = prediction(plant_name, file_path)
            if diseases == None:
                diseases_info="Diseases are not trained for this Plant"
                return render_template('disease_result.html',cond=1, plant=plant_name, status=plant_status, diseases_name=diseases_info)

            else:
                disease = str(diseases)
                index_value = diseases_preventation.Diseases_Name[diseases_preventation.Diseases_Name == str(diseases)].index.tolist()
                precaution = diseases_preventation.iloc[index_value[0]]["Precautions"]
#               return render_template('disease.html', plant=plant_name, status=plant_status, diseases_name=disease,disease_precautions=precaution)
                return render_template('disease_result.html',cond=2,plant=plant_name,status=plant_status, diseases_name=disease,disease_precautions=precaution)



if __name__ == '__main__':
    app.run(debug=False)



