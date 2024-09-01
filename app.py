from flask import Flask, render_template, request
import numpy as np
import pickle
import logging

# Load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Example fraud and non-fraud data (remove the last label value when using)
example_non_fraud = [
-0.208428372, 1.072131713, -0.281689118, 1.188254291, 1.739633653, -0.848113627, 0.564671891, -0.633982292, 0.371499978, 1.484734564, -0.372933052, -1.370381438, 0.161307438, -1.810232843, -0.434540611, -1.53390402, -1.073780244, 0.862769022, -1.16375028, 0.183810584, -0.301033763, -0.431254088, -0.847681421, 0.100169974, 0.03947497, 0.372361233, -0.504906488, 0.080580063, 0.026028918, -0.297256186, 0.704137811, -0.276624458, 0.717183338, -0.116790314, -0.84908629, 0.86992873, -1.172601746, 1.081514085, -0.105476561, -0.29684094, -1.209014057, 1.114980991, 0.15992316, -0.834077052, -0.551104588, -0.445583043, 0.578707279, -0.048044627, 1.036923936, -0.989675985, -0.148633715, 0.344637459, 0.897329008
]

example_fraud = [
-1.115287857, -19.13973286, 9.28684736, -20.1349921, 7.81867331, -15.65220768, -1.668347707, -21.3404781, 0.641899701, -8.550110327, -16.64962816, 4.818152447, -9.445314783, 1.317056293, -7.243460974, 0.830910291, -9.53325705, -18.75064115, -8.092648773, 3.326758275, 0.427203431, -2.182691946, 0.520543072, -0.760556415, 0.662766638, -0.948454306, 0.121795926, -3.381842929, -1.256523621, 0.206102866, -1.037584119, -1.519243531, 2.308492454, -1.50359932, 2.064100584, -1.000845058, -1.016897033, -2.05973052, -0.275166375, -1.562206097, -2.755796928, 3.43824826, -3.52152936, -0.918761288, -4.452100249, 0.499313754, -2.907903434, -5.248646395, -0.936814653, 1.160120294, 0.175018552, 1.307871432, 0.102825644, -0.017746283
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input from the form
            me = request.form['message']
            # Split the input by spaces and convert to float
            message = [float(x) for x in me.split()]
            num_inputs = len(message)

            # Log the input message
            logging.debug(f"Input message: {message}")

            # Ensure the input has the required number of features (30)
            if num_inputs != 30:
                return render_template('home.html', error=f"Please provide exactly 30 feature values. You provided {num_inputs}.")
            
            # Reshape the input for prediction
            vect = np.array(message).reshape(1, -1)
            # Make prediction
            my_prediction = clf.predict(vect)
            prediction_proba = clf.predict_proba(vect)[0]

            # Log the prediction and probabilities
            logging.debug(f"Prediction: {my_prediction[0]}")
            logging.debug(f"Prediction Probabilities: {prediction_proba}")

            return render_template('result.html', prediction=my_prediction[0], prediction_proba=prediction_proba)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template('home.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
