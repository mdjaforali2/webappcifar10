# views.py
import os
import uuid
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from tensorflow import keras
from PIL import Image
from django.contrib import messages
from .models import ImagePrediction
import json
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from tensorflow import keras
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from .models import ImagePrediction
import json



# Assuming 'classes' is a global variable
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the pre-trained model
plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')


model_path = os.path.join(settings.BASE_DIR, 'models/cifar10_model.h5')
model = keras.models.load_model(model_path)


def predict_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Make prediction
    raw_predictions = model.predict(img_batch)
    predicted_class = np.argmax(raw_predictions)

    # Convert probabilities to a standard Python data type (list)
    flat_predictions = raw_predictions.flatten()
    probabilities = (np.exp(flat_predictions) / np.sum(np.exp(flat_predictions))).tolist()

    # Get the class name
    predicted_class_name = classes[predicted_class]

    return predicted_class_name, probabilities


import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_probabilities(class_names, probabilities):
    fig, ax = plt.subplots()
    ax.bar(class_names, probabilities, color='blue')
    ax.set_ylabel('Probability')
    ax.set_title('Class Probabilities')

    # Set the tick positions
    ax.set_xticks(range(len(class_names)))

    # Set the tick labels with rotation
    ax.set_xticklabels(class_names, rotation=45, ha='right')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Convert the plot to a base64-encoded string
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()  # Close the plot to release resources

    return plot_base64



def upload_image(request):
    if request.method == 'POST':
        # Handle image upload
        uploaded_file = request.FILES['image']
        unique_filename = str(uuid.uuid4()) + '_' + uploaded_file.name
        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)

        with open(upload_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Make prediction
        predicted_class, probabilities = predict_image(upload_path)

        # Save the prediction to the database
        ImagePrediction.objects.create(
            image=upload_path,
            predicted_class=predicted_class,
            probabilities=probabilities
        )

        # Redirect to a page displaying the prediction results
        return redirect('prediction_results')

    return render(request, 'upload_image.html')

def prediction_results(request):
    # Retrieve the latest prediction from the database
    latest_prediction = ImagePrediction.objects.latest('id')

    context = {
        'image_path': latest_prediction.image.url,
        'predicted_class': latest_prediction.predicted_class,
        'probabilities': latest_prediction.probabilities,
        'plot_base64': plot_probabilities(classes, latest_prediction.probabilities),
    }

    return render(request, 'prediction_results.html', context)





# remaining views and functions...
from django.shortcuts import render, redirect
from .models import ImagePrediction

def clear_uploads(request):
    if request.method == 'POST':
        # Delete all recent uploads
        ImagePrediction.objects.all().delete()
        return redirect('upload_image')  # Redirect back to the main upload page
    return render(request, 'clear_uploads.html')


def home(request):
    accloss_plot_path = 'accloss.png'  # Replace with the actual path

    context = {
        'accuracy_plot_path': accloss_plot_path,
    }

    return render(request, 'home.html', context)