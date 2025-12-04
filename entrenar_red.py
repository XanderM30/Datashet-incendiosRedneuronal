import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os

# --- Configuración ---
DATASET_DIR = r"C:\Datashet incendios\dataset_final"
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 50
MODEL_SAVE_PATH = r"C:\Datashet incendios\assets\models\modelo_incendios_mejorado.h5"
TFLITE_SAVE_PATH = r"C:\Datashet incendios\assets\models\modelo_incendios.tflite"

# Crear carpeta si no existe
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- Generadores de datos con aumentos agresivos ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5,1.5]
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Clases:", train_generator.class_indices)

# --- Calcular class weights ---
y_train = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# --- Modelo con Transfer Learning ---
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Congelar todas las capas primero
base_model.trainable = False

# Descongelar últimas 20 capas para fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# --- Compilación con learning rate bajo para fine-tuning ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Early stopping ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Entrenamiento ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# --- Evaluación ---
loss, acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# --- Guardar modelo Keras ---
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Modelo guardado en {MODEL_SAVE_PATH}")

# --- Conversión a TFLite (sobrescribiendo si existe) ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_SAVE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"\n✅ Modelo TFLite guardado en {TFLITE_SAVE_PATH}")
print("\nProceso completado.")
print(" git add .")
print(" git commit -m 'Modelo de incendios actualizado'")
print(" git push origin main")
print("\nRecuerda actualizar el modelo en la aplicación móvil si es necesario.")
print("¡Gracias por usar este script!")
print("👋😊")
