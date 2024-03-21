from flask import Flask,jsonify

from speechbrain.pretrained import SpeakerRecognition
import json

file = json.load(open("data.json"))

app =  Flask(__name__)
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

@app.get("/")
def index():
    r = []
    for i in file:
        print(i)
        score, prediction = verification.verify_files(i['file'], "main2.wav")
        score, prediction = score[0].item() , prediction[0].item()
        
        r.append({"value":prediction,"name":i['name']})

    return jsonify(r)

