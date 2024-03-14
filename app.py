import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

with open ('model_pred.pkl','rb') as f:
    (vectorizer,model)=pickle.load(f)

@app.route('/',methods=['GET'])
def route():
    return "Hello"

@app.route('/emotions',methods =['POST'])
def emotions():
    request_text = request.get_json()

    text=vectorizer.transform([request_text])
    prediction=model.predict(text)

    if (prediction[0]==0):
        message='Are you experiencing Sadness..?'
    elif (prediction[0]==1):
        message='Seems joyous and happy'
    elif (prediction[0]==2):
        message='It feels like Love'
    elif (prediction[0]==3):
        message='Are you experiencing Anger?'
    elif (prediction[0]==4):
        message='Comes across as fearful'
    elif (prediction[0]==5):
        message='Surprise'

    return jsonify({"emotion": message}), 200

if __name__ == '__main__':
    app.run(debug=True)