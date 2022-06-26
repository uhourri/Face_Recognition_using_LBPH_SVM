from model.tools import draw_in_image
import cv2
import io
import base64
from PIL import Image
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    path = 'tmp/'+uploaded_file.filename
    uploaded_file.save(path)
    img = draw_in_image(path)
    cv2.imwrite(path, img)
    image = Image.open(path, 'r')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    encoded_image = base64.encodebytes(image_bytes.getvalue()).decode('ascii')
    return jsonify(encoded_image)


@app.route("/")
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()