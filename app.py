from torchvision import models
import torch
from flask import Flask, render_template,request, redirect
from PIL import Image
from torchvision import transforms
import os
from pysentimiento import create_analyzer


model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.eval()

# model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
# model.eval()

app = Flask(__name__)
analyzer = create_analyzer(task="sentiment", lang="en")
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def main():
    return render_template("index.html")

@app.route("/clear")
def clear():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(img_path)
        file.save(img_path)
        
        input_image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)

        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        
        return render_template("index.html", categories=categories, top5_catid=top5_catid, top5prob=top5_prob, name=file.filename)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = analyzer.predict(text)
    probabilities = result.probas
    probabilities = {k: round(v * 100, 2) for k, v in probabilities.items()}
    return render_template('index.html', text=text, result=result, probabilities=probabilities)



if __name__ == '__main__':
    app.run(debug=True)
