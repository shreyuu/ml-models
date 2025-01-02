# Posture Recognition: CNN-Based Sitting and Standing Detection

This Python script demonstrates how to build and train a Convolutional Neural Network (CNN) for classifying images into two categories: "sitting" and "standing" from a dataset of images.

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib

You can install the required dependencies using pip:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Dataset

The dataset used in this script is assumed to be organized as follows:

```
sitting-standing-dataset-images/
    └── train/
        ├── _annotations.csv
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

- `_annotations.csv`: A CSV file containing image filenames and their corresponding class labels (0 for "sitting", 1 for "standing").
- Images: Each image should correspond to an entry in the CSV file.

## Script Overview

1. **Data Loading & Preprocessing**: 
   - Loads images from the dataset and resizes them to `224x224` pixels.
   - Normalizes image pixel values and converts labels to categorical format.

2. **Model Definition**: 
   - A CNN model is built using TensorFlow's Keras API, with multiple convolutional layers followed by dense layers for classification.

3. **Data Augmentation**: 
   - To prevent overfitting, data augmentation techniques like rotation, width and height shift, and horizontal flip are applied.

4. **Model Training**: 
   - The model is trained using the augmented data with a validation split from the test set.

5. **Model Evaluation**: 
   - After training, the model is evaluated on the test set, and the accuracy is printed.

6. **Plotting Results**: 
   - The training and validation accuracy and loss are plotted for visual inspection.

7. **Image Prediction**: 
   - A function `predict_image` is provided to classify new images.

## Code Breakdown

### 1. Import Libraries

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
```

### 2. Load and Preprocess Data

The `load_data` function loads images and their corresponding labels from the CSV file.

```python
def load_data(csv_path, img_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for idx, row in df.iterrows():
        img = load_img(os.path.join(img_path, row['filename']), target_size=(224, 224))
        img_array = img_to_array(img)
        X.append(img_array)
        y.append(row['class'])
    return np.array(X), np.array(y)
```

### 3. Label Encoding and One-Hot Encoding

The labels are encoded using `LabelEncoder` and then converted to categorical format using `to_categorical`.

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
```

### 4. Train-Test Split

The dataset is split into training and testing sets (80% train, 20% test).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
```

### 5. Model Definition

A CNN model is built using three convolutional layers followed by dense layers.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### 6. Compile and Train the Model

The model is compiled with the Adam optimizer and categorical crossentropy loss. The training is performed with augmented data.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=(X_test, y_test)
)
```

### 7. Model Evaluation

After training, the model is evaluated on the test set.

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

### 8. Plot Training History

The training and validation accuracy and loss are plotted for visual inspection.

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

### 9. Image Prediction

A function `predict_image` is defined to predict the class of a new image.

```python
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {np.max(prediction):.2f}")
```

### 10. Example Usage

You can use the following to predict a new image:

```python
predict_image('sitting-standing-dataset-images/train/000127_jpg.rf.f340495dc664cbbeeea7ad5cb238c523.jpg')
```

### 11. Classes

- **Class 0**: Sitting
- **Class 1**: Standing

## Model Results

The script will output the following:

- Test accuracy after evaluation.
- A plot showing the training and validation accuracy and loss over epochs.
- A prediction for the new image with class and confidence.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.