from flask import Flask, request, redirect, render_template
from PIL import Image
from glob import glob
from uuid import uuid4
from pathlib import Path
import numpy as np

from feature_extractor import FeatureExtractor

app = Flask(__name__)

fe = FeatureExtractor()

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload():
  return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
  files = request.files.getlist("query_images")

  Path("static/images").mkdir(parents=True, exist_ok=True)
  Path("static/features").mkdir(parents=True, exist_ok=True)

  for file in files:
    if file:
      image = Image.open(file).convert('RGB')
      feature = fe(image)

      name = uuid4().hex

      image.save(f'static/images/{name}.jpg')
      np.save(f'static/features/{name}.npy', feature)

  return render_template('upload.html')

@app.route('/search', methods=['POST'])
def search_image():
  file = request.files['query_image']

  if file:
    image = Image.open(file).convert('RGB')
    feature = fe(image)

    name = uuid4().hex

    images = np.array(glob('static/images/*.jpg'))
    features = np.array([np.load(feature) for feature in glob('static/features/*.npy')])

    if len(images) == 0:
      return redirect('/')

    scores = np.linalg.norm(features - feature, axis=1)
    indices = np.argsort(scores)

    k = min(len(scores), 30)

    if scores[indices[0]] == 0:
      return render_template('index.html', images=images[indices][:k])
    else:
      image.save(f'static/images/{name}.jpg')
      np.save(f'static/features/{name}.npy', feature)

      return render_template('index.html', images=images[indices][:k])
  else:
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)