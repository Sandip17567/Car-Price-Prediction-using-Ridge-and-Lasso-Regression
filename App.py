import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open("ridge_model.pkl", "rb"))
Scaler=pickle.load(open("X_Scaler.pkl", "rb"))
y_scale=pickle.load(open("y_Scaler.pkl", "rb"))
model_columns=pickle.load(open("columns.pkl", "rb"))


st.title("Car Price Prediction App")
# Input fields for user to enter car features
symboling = st.select_slider('Symboling', options=[-3, -2, -1, 0, 1, 2, 3])
carname=st.selectbox("Car Name",["alfa-romero","audi","bmw","buick","chevrolet","dodge","honda","isuzu","jaguar",
"mazda","mercury","mitsubishi","nissan","peugeot","plymouth","porsche","renault","saab","subaru","toyota","volkswagen","volvo"
])
fueltype=st.selectbox('Fuel Type', ['gas', 'diesel'])
aspiration = st.selectbox("Aspiration",["std","turbo"])
doornumber = st.selectbox("Door Number",["two","four"])
carbody = st.selectbox("Car Body",["sedan","hatchback","wagon","hardtop","convertible"])
drivewheel = st.selectbox("Drive Wheel",["fwd","rwd","4wd"])
enginelocation = st.selectbox("Engine Location",["front","rear"])
wheelbase = st.number_input("Wheelbase")
carlength = st.number_input("Car Length")
carwidth = st.number_input("Car Width")
carheight = st.number_input("Car Height")
curbweight = st.number_input("Curb Weight")
enginetype = st.selectbox("Engine Type",
["dohc","ohcv","ohc","l","rotor","ohcf","dohcv"])
cylindernumber = st.selectbox("Cylinder Number",
["two","three","four","five","six","eight","twelve"])
enginesize = st.number_input("Engine Size")
fuelsystem = st.selectbox("Fuel System",
["mpfi","2bbl","idi","1bbl","spdi","4bbl","spfi"])
boreratio = st.number_input("Bore Ratio")
stroke = st.number_input("Stroke")
compressionratio = st.number_input("Compression Ratio")

horsepower = st.number_input("Horsepower")
peakrpm = st.number_input("Peak RPM")

citympg = st.number_input("City MPG")
highwaympg = st.number_input("Highway MPG")

if st.button("Predict Price"):

    input_dict = {
        "symboling":symboling,
        "carname":carname,
        "fueltype":fueltype,
        "aspiration":aspiration,
        "doornumber":doornumber,
        "carbody":carbody,
        "drivewheel":drivewheel,
        "enginelocation":enginelocation,
        "wheelbase":wheelbase,
        "carlength":carlength,
        "carwidth":carwidth,
        "carheight":carheight,
        "curbweight":curbweight,
        "enginetype":enginetype,
        "cylindernumber":cylindernumber,
        "enginesize":enginesize,
        "fuelsystem":fuelsystem,
        "boreratio":boreratio,
        "stroke":stroke,
        "compressionratio":compressionratio,
        "horsepower":horsepower,
        "peakrpm":peakrpm,
        "citympg":citympg,
        "highwaympg":highwaympg
    }



    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df).astype(float)

    for col in model_columns:
        if col not in input_df.columns:
         input_df[col] = 0

    input_df = input_df[model_columns]

    input_scaled = Scaler.transform(input_df)
    predict_scaled= model.predict(input_scaled)
    prediction_actual=y_scale.inverse_transform(predict_scaled.reshape(-1, 1))

    st.success(f"Predicted Car Price: {prediction_actual[0].item():.2f}")