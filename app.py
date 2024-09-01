from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model, label encoder, and scaler
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_and_predict(input_data):
    # Encode the player names
    input_data['Player1_Encoded'] = label_encoder.transform(input_data['Player1_Int'])
    input_data['Player2_Encoded'] = label_encoder.transform(input_data['Player2_Int'])

    # Select features and apply scaler
    features = ['Player1_Rank', 'Player2_Rank', 'Player1_Odds', 'Player2_Odds', 'Player1_Implied_Prob', 'Player2_Implied_Prob', 'Player1_Encoded', 'Player2_Encoded']
    input_data = input_data[features]  # Ensure only required columns are present
    scaled_features = scaler.transform(input_data)
    
    # Combine the scaled features into a DataFrame
    input_data_preprocessed = pd.DataFrame(scaled_features, columns=features)

    # Make prediction
    prediction = model.predict(input_data_preprocessed)
    return prediction

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        player1 = request.form['player1_name']
        player2 = request.form['player2_name']
        player1_rank = float(request.form['player1_rank'])
        player2_rank = float(request.form['player2_rank'])
        player1_odds = float(request.form['player1_odds'])
        player2_odds = float(request.form['player2_odds'])
        player1_implied_prob = float(request.form['player1_implied_prob'])
        player2_implied_prob = float(request.form['player2_implied_prob'])

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'Player1_Int': [player1],
            'Player2_Int': [player2],
            'Player1_Rank': [player1_rank],
            'Player2_Rank': [player2_rank],
            'Player1_Odds': [player1_odds],
            'Player2_Odds': [player2_odds],
            'Player1_Implied_Prob': [player1_implied_prob],
            'Player2_Implied_Prob': [player2_implied_prob]
        })

        # Preprocess and predict
        prediction = preprocess_and_predict(input_data)
        predicted_winner =player1 if prediction[0] == 1 else player2

        # Return the result as a response
        return render_template('result.html', player1=player1, player2=player2, prediction_text=f'Predicted Winner: {predicted_winner}')
    
    except Exception as e:
        # Handle errors and return a message
        return render_template('result.html', player1=None, player2=None, prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
