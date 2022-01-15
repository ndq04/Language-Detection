import cv2
import easyocr
from flask import Flask, request
from flask_cors import CORS, cross_origin
from langdetect import detect

app = Flask(__name__)
CORS(app, resources={r"/api/*":{"origins":"*"}})
app.config["CORS HEADERS"] = "Content-Type"

@app.route("/")
@cross_origin()
def Home():
  return str("Welcome Home")

@app.route("/api/language_detection", methods=['POST'])
@cross_origin()
def languageDetection():
  image_path  = request.json["image_path"]

  reader = easyocr.Reader(['en','vi'])
  reader = easyocr.Reader(['en','ja'])
  reader = easyocr.Reader(['en','ch_tra'])
  reader = easyocr.Reader(['en','ch_sim'])

  results = reader.readtext(image_path)

  text = ''
  for result in results:
    text += result[1] + ''
  return detect(text)

if __name__ == "__main__":
  app.run(debug=True)
