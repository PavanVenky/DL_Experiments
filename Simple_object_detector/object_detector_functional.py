"""
Simple Two-Class Object Detector using VGG16 (Functional Version)
Detects cars and trucks with bounding box regression

Author: Claude
Date: January 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def build_object_detector(input_shape=(224, 224, 3), num_classes=2):
    """
    Build the object detection model with VGG16 backbone
    
    Args:
        input_shape: tuple, input image shape (height, width, channels)
        num_classes: int, number of classes (2 for car and truck)
        
    Returns:
        keras Model with two outputs: classification and bounding box
    """
    # Load pretrained VGG16 without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers of VGG16 (optional - can fine-tune later)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # VGG16 feature extraction
    x = base_model(inputs, training=False)
    
    # Flatten features
    x = layers.Flatten()(x)
    
    # Shared dense layers
    x = layers.Dense(512, activation='relu', name='shared_dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', name='shared_dense_2')(x)
    x = layers.Dropout(0.5)(x)
    
    # Classification head (car vs truck)
    # Output: probability distribution over classes
    class_output = layers.Dense(128, activation='relu', name='class_dense')(x)
    class_output = layers.Dense(
        num_classes, 
        activation='softmax', 
        name='classification'
    )(class_output)
    
    # Bounding box regression head
    # Output: 4 values (x_min, y_min, x_max, y_max)
    bbox_output = layers.Dense(128, activation='relu', name='bbox_dense')(x)
    bbox_output = layers.Dense(
        4, 
        activation='linear',  # Linear activation for regression
        name='bounding_box'
    )(bbox_output)
    
    # Create model with two outputs
    model = models.Model(
        inputs=inputs,
        outputs=[class_output, bbox_output],
        name='simple_object_detector'
    )
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate losses for each head
    
    Args:
        model: keras Model
        learning_rate: float, learning rate for optimizer
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'classification': 'categorical_crossentropy',
            'bounding_box': 'mse'  # Mean squared error for bbox regression
        },
        loss_weights={
            'classification': 1.0,
            'bounding_box': 1.0  # Adjust these weights based on performance
        },
        metrics={
            'classification': 'accuracy',
            'bounding_box': 'mae'  # Mean absolute error
        }
    )
    
    return model


def prepare_data(images, annotations, num_classes=2):
    """
    Prepare data for training
    
    Args:
        images: numpy array of images, shape (N, H, W, 3)
        annotations: list of tuples (x_min, y_min, x_max, y_max, class_id)
        num_classes: number of classes (2 for car and truck)
        
    Returns:
        images, class_labels, bbox_labels
    """
    bbox_labels = []
    class_labels = []
    
    for annotation in annotations:
        x_min, y_min, x_max, y_max, class_id = annotation
        
        # Bounding box coordinates
        bbox_labels.append([x_min, y_min, x_max, y_max])
        
        # One-hot encode class labels
        class_label = np.zeros(num_classes)
        class_label[class_id] = 1
        class_labels.append(class_label)
    
    bbox_labels = np.array(bbox_labels, dtype=np.float32)
    class_labels = np.array(class_labels, dtype=np.float32)
    
    # Normalize bounding box coordinates (0-1 range)
    # Assuming images are all same size
    if len(images) > 0:
        img_height, img_width = images[0].shape[:2]
        bbox_labels[:, [0, 2]] /= img_width  # Normalize x coordinates
        bbox_labels[:, [1, 3]] /= img_height  # Normalize y coordinates
    
    return images, class_labels, bbox_labels


def preprocess_images(images):
    """
    Preprocess images for VGG16
    
    Args:
        images: numpy array of images
        
    Returns:
        preprocessed images
    """
    return tf.keras.applications.vgg16.preprocess_input(images)


def train_model(model, X_train, y_class_train, y_bbox_train,
                X_val=None, y_class_val=None, y_bbox_val=None,
                epochs=20, batch_size=16):
    """
    Train the model
    
    Args:
        model: keras Model
        X_train: training images
        y_class_train: training class labels
        y_bbox_train: training bounding box labels
        X_val: validation images (optional)
        y_class_val: validation class labels (optional)
        y_bbox_val: validation bounding box labels (optional)
        epochs: number of training epochs
        batch_size: batch size for training
        
    Returns:
        training history
    """
    validation_data = None
    if X_val is not None:
        validation_data = (
            X_val,
            {
                'classification': y_class_val,
                'bounding_box': y_bbox_val
            }
        )
    
    history = model.fit(
        X_train,
        {
            'classification': y_class_train,
            'bounding_box': y_bbox_train
        },
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history


def predict_single_image(model, image):
    """
    Make predictions on a single image
    
    Args:
        model: trained keras Model
        image: preprocessed image, shape (H, W, 3)
        
    Returns:
        class_prediction, bbox_prediction
    """
    # Add batch dimension
    img = np.expand_dims(image, axis=0)
    
    # Preprocess image
    img = tf.keras.applications.vgg16.preprocess_input(img)
    
    # Make prediction
    class_pred, bbox_pred = model.predict(img, verbose=0)
    
    return class_pred, bbox_pred


def predict_batch(model, images):
    """
    Make predictions on a batch of images
    
    Args:
        model: trained keras Model
        images: preprocessed images, shape (N, H, W, 3)
        
    Returns:
        class_predictions, bbox_predictions
    """
    # Preprocess images
    imgs = tf.keras.applications.vgg16.preprocess_input(images)
    
    # Make predictions
    class_preds, bbox_preds = model.predict(imgs, verbose=0)
    
    return class_preds, bbox_preds


def denormalize_bbox(bbox, img_width, img_height):
    """
    Denormalize bounding box coordinates back to pixel values
    
    Args:
        bbox: normalized bounding box [x_min, y_min, x_max, y_max]
        img_width: image width in pixels
        img_height: image height in pixels
        
    Returns:
        denormalized bounding box
    """
    bbox_denorm = bbox.copy()
    bbox_denorm[0] *= img_width
    bbox_denorm[2] *= img_width
    bbox_denorm[1] *= img_height
    bbox_denorm[3] *= img_height
    return bbox_denorm.astype(int)


def visualize_prediction(class_pred, bbox_pred, class_names=['Car', 'Truck']):
    """
    Print prediction results
    
    Args:
        class_pred: class prediction array
        bbox_pred: bounding box prediction array
        class_names: list of class names
    """
    # Get predicted class
    predicted_class_idx = np.argmax(class_pred[0])
    class_name = class_names[predicted_class_idx]
    confidence = class_pred[0][predicted_class_idx]
    
    # Get bounding box
    bbox = bbox_pred[0]
    
    print(f"Predicted Class: {class_name} (confidence: {confidence:.4f})")
    print(f"Bounding Box (normalized): [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")


def save_model(model, filepath):
    """
    Save the model to disk
    
    Args:
        model: keras Model
        filepath: path to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a saved model from disk
    
    Args:
        filepath: path to the saved model
        
    Returns:
        loaded keras Model
    """
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model


def generate_dummy_data(num_samples=100, img_size=224):
    """
    Generate dummy data for testing
    
    Args:
        num_samples: number of samples to generate
        img_size: size of images
        
    Returns:
        images, annotations
    """
    # Generate random images
    images = np.random.rand(num_samples, img_size, img_size, 3).astype(np.float32)
    
    # Generate random annotations
    annotations = []
    for i in range(num_samples):
        x_min = np.random.randint(0, 100)
        y_min = np.random.randint(0, 100)
        x_max = np.random.randint(x_min + 50, 200)
        y_max = np.random.randint(y_min + 50, 200)
        class_id = np.random.randint(0, 2)  # 0 for car, 1 for truck
        annotations.append((x_min, y_min, x_max, y_max, class_id))
    
    return images, annotations


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Simple Two-Class Object Detector Training Pipeline")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n[1/10] Generating dummy data...")
    print("NOTE: Replace this with your actual data!")
    num_samples = 100
    images, annotations = generate_dummy_data(num_samples=num_samples)
    
    # Step 2: Prepare data
    print("[2/10] Preparing data...")
    images, class_labels, bbox_labels = prepare_data(images, annotations, num_classes=2)
    
    # Step 3: Preprocess images
    print("[3/10] Preprocessing images...")
    images = preprocess_images(images)
    
    # Step 4: Split data
    print("[4/10] Splitting data...")
    (X_train, X_val, 
     y_class_train, y_class_val,
     y_bbox_train, y_bbox_val) = train_test_split(
        images, class_labels, bbox_labels,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Step 5: Build model
    print("\n[5/10] Building model...")
    model = build_object_detector(input_shape=(224, 224, 3), num_classes=2)
    
    # Step 6: Compile model
    print("[6/10] Compiling model...")
    model = compile_model(model, learning_rate=0.0001)
    
    # Step 7: Display model summary
    print("[7/10] Model summary:")
    model.summary()
    
    # Step 8: Train model
    print("\n[8/10] Training model...")
    history = train_model(
        model,
        X_train, y_class_train, y_bbox_train,
        X_val, y_class_val, y_bbox_val,
        epochs=5,  # Increase for real training
        batch_size=16
    )
    
    # Step 9: Save model
    print("\n[9/10] Saving model...")
    save_model(model, 'object_detector_model.h5')
    
    # Step 10: Test prediction
    print("\n[10/10] Testing prediction...")
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    class_pred, bbox_pred = predict_single_image(model, test_image)
    visualize_prediction(class_pred, bbox_pred)
    
    # Print denormalized bbox
    bbox_denorm = denormalize_bbox(bbox_pred[0], 224, 224)
    print(f"Bounding Box (pixels): {bbox_denorm}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return model, history


# Example of how to use the trained model
def example_inference():
    """
    Example of loading and using a trained model
    """
    print("\n" + "=" * 60)
    print("Example Inference")
    print("=" * 60)
    
    # Load the trained model
    model = load_model('object_detector_model.h5')
    
    # Prepare test image (replace with actual image)
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Make prediction
    class_pred, bbox_pred = predict_single_image(model, test_image)
    
    # Visualize results
    visualize_prediction(class_pred, bbox_pred)
    
    # Denormalize bbox for visualization
    bbox_denorm = denormalize_bbox(bbox_pred[0], 224, 224)
    print(f"Bounding Box (pixels): {bbox_denorm}")


if __name__ == "__main__":
    # Run the training pipeline
    model, history = main()
    
    print("\n" + "=" * 60)
    print("How to use this code with your own data:")
    print("=" * 60)
    print("""
1. Load your images as numpy arrays (shape: N x 224 x 224 x 3)
2. Prepare annotations as list of tuples: (x_min, y_min, x_max, y_max, class_id)
   - class_id: 0 for car, 1 for truck
3. Replace generate_dummy_data() with your actual data loading
4. Adjust hyperparameters (epochs, batch_size, learning_rate) as needed
5. Train and save the model

Example Usage:
    # Training
    images, annotations = load_your_data()  # Your data loading function
    images, class_labels, bbox_labels = prepare_data(images, annotations)
    images = preprocess_images(images)
    model = build_object_detector()
    model = compile_model(model)
    history = train_model(model, images, class_labels, bbox_labels, epochs=20)
    save_model(model, 'my_model.h5')
    
    # Inference
    model = load_model('my_model.h5')
    class_pred, bbox_pred = predict_single_image(model, your_image)
    visualize_prediction(class_pred, bbox_pred)
    """)
    
    # Uncomment to test inference
    # example_inference()