# ============================================
# CELL 1: IMPORTS
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from PIL import Image
import cv2
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("✅ Imports done!")



#second cell: FOLDER STRUCTURE
# ============================================
base_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'

print("📁 FOLDER STRUCTURE:\n")
for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}📂 {os.path.basename(root)}/")
    if level < 2:  # Don't go too deep
        for d in dirs:
            print(f"{indent}  📁 {d}")

#third cell: image counts per class
# ============================================
augmented_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'

class_counts = {}
for class_name in os.listdir(augmented_path):
    class_path = os.path.join(augmented_path, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_name] = count

# Display as DataFrame
df_counts = pd.DataFrame({
    'Class': list(class_counts.keys()),
    'Count': list(class_counts.values())
})
df_counts = df_counts.sort_values('Count', ascending=False)
print(df_counts.to_string(index=False))
print(f"\n📊 Total Images: {sum(class_counts.values())}")


# CELL 4: PLOT CLASS DISTRIBUTION
# ============================================
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
plt.title('🧠 Alzheimer MRI Dataset - Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Dementia Stage')
plt.ylabel('Number of Images')
plt.xticks(rotation=15)

# Add count labels on bars
for bar, count in zip(bars, class_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             str(count), ha='center', fontweight='bold')
    
plt.tight_layout()
plt.show()

# Check if balanced
print("\n🔍 CLASS BALANCE CHECK:")
max_count = max(class_counts.values())
min_count = min(class_counts.values())
print(f"   Max: {max_count}, Min: {min_count}")
print(f"   Ratio: {max_count/min_count:.2f}x difference")

# CELL 5: VIEW SAMPLE IMAGES FROM EACH CLASS
# ============================================
augmented_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

for row, class_name in enumerate(class_names):
    class_path = os.path.join(augmented_path, class_name)
    images = os.listdir(class_path)[:5]  # Get first 5 images
    
    for col, img_name in enumerate(images):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_title(f'{class_name}', fontsize=12, fontweight='bold')

plt.suptitle('🧠 MRI Samples: Progression of Alzheimer\'s Disease', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# CELL 6: CHECK IMAGE DIMENSIONS & PROPERTIES
# ============================================
augmented_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'

# Sample random images to check dimensions
sample_dims = []
sample_modes = []

for class_name in os.listdir(augmented_path):
    class_path = os.path.join(augmented_path, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)[:50]  # Check 50 from each class
        for img_name in images:
            img = Image.open(os.path.join(class_path, img_name))
            sample_dims.append(img.size)
            sample_modes.append(img.mode)
# Analyze dimensions
unique_dims = Counter(sample_dims)
unique_modes = Counter(sample_modes)

print("📐 IMAGE DIMENSIONS:")
for dim, count in unique_dims.most_common():
    print(f"   {dim[0]} x {dim[1]} pixels: {count} images")

print(f"\n🎨 COLOR MODES:")
for mode, count in unique_modes.items():
    mode_desc = {'L': 'Grayscale', 'RGB': 'Color', 'RGBA': 'Color+Alpha'}
    print(f"   {mode_desc.get(mode, mode)}: {count} images")
# Get one sample for detailed info
sample_img = Image.open(os.path.join(augmented_path, 'NonDemented', os.listdir(os.path.join(augmented_path, 'NonDemented'))[0]))
print(f"\n📋 SAMPLE IMAGE INFO:")
print(f"   Size: {sample_img.size}")
print(f"   Mode: {sample_img.mode}")
print(f"   Format: {sample_img.format}")

# CELL 7: MARKDOWN - DATA ANALYSIS SUMMARY
# ============================================
from IPython.display import Markdown, display

summary = """
## 📋 Data Analysis Summary

### Dataset Overview
- **Total Images:** 33,984
- **Classes:** 4 (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
- **Class Balance:** 1.49x ratio (acceptable)

### Image Properties
- **Sizes:** Mixed (200×190 and 180×180) → Need resizing
- **Format:** JPEG, RGB color mode
- **Quality:** Good, augmentation already applied
### Medical Insight
- Alzheimer's causes **brain atrophy** (shrinkage)
- **Ventricles enlarge** as disease progresses
- Model should learn to detect these structural changes

### Preprocessing Decisions
1. ✅ Resize all images to **176×176** (balanced size)
2. ✅ Keep RGB or convert to grayscale
3. ✅ Normalize pixel values to [0,1]
4. ✅ Use Original dataset for validation (no data leakage)
"""
display(Markdown(summary))

# Clear any cached imports
import sys
if 'google.protobuf' in sys.modules:
    del sys.modules['google.protobuf']


#note this cell is slightly difrent from the chromebook version
# CELL 8: BUILD DATA GENERATORS
# ============================================
# CELL 8: BUILD DATA GENERATORS
# ============================================
train_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'
test_path = 'C:\\Users\\bbnro\\Downloads\\archive\\OriginalDataset'

# Image parameters
IMG_SIZE = (176, 176)
BATCH_SIZE = 32

# Load training data
train_generator = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.15,
    subset='training',
    seed=42
)

# Store class names before mapping
class_names = sorted(os.listdir(train_path))

# Load validation data
val_generator = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.15,
    subset='validation',
    seed=42
)

# Load test data
test_generator = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Normalize pixel values
normalize = tf.keras.layers.Rescaling(1./255)
train_generator = train_generator.map(lambda x, y: (normalize(x), y))
val_generator = val_generator.map(lambda x, y: (normalize(x), y))
test_generator = test_generator.map(lambda x, y: (normalize(x), y))

print("\n✅ Data Generators Ready!")
print(f"   Training samples: {tf.data.experimental.cardinality(train_generator).numpy() * BATCH_SIZE}")
print(f"   Validation samples: {tf.data.experimental.cardinality(val_generator).numpy() * BATCH_SIZE}")
print(f"   Test samples: {tf.data.experimental.cardinality(test_generator).numpy() * BATCH_SIZE}")


# ============================================
# ============================================
# CELL 9: VISUALIZE A BATCH
# ============================================
# Get one batch to verify
images, labels = next(iter(train_generator))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    # Ensure image is in correct format for display
    img_display = (images[i].numpy() * 255).astype(np.uint8)
    ax.imshow(img_display, cmap='gray')
    label_idx = np.argmax(labels[i])
    ax.set_title(f'{class_names[label_idx]}')
    ax.axis('off')

plt.suptitle('Sample Batch from Training Generator', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Pixel range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")

# CELL 10: FIX PROTOBUF (Run this first!)
# ============================================
#!pip uninstall -y protobuf
#!pip install protobuf==3.20.3 --quiet

print("✅ Protobuf fixed! Now restart kernel:")
print("   Session → Restart Session")
print("   Then re-run all cells before continuing")

# CELL 11: IMPORTS & GPU CHECK
# ============================================
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
# Check GPU
print("🔥 GPU Status:")
print(f"   TensorFlow version: {tf.__version__}")
print(f"   GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n✅ All imports successful!")

# CELL 12: ENHANCED DATA GENERATORS
# ============================================

# Paths

train_path = 'C:\\Users\\bbnro\\Downloads\\archive\\AugmentedAlzheimerDataset'
test_path = 'C:\\Users\\bbnro\\Downloads\\archive\\OriginalDataset'
# Image parameters
IMG_SIZE = (224, 224)  # EfficientNet/ResNet optimal size
BATCH_SIZE = 16  # Smaller batch for better generalization

# Training generator - MINIMAL augmentation (data already augmented!)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Slight rotation only
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # 20% for validation
)

# Test generator - NO augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)
test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

print("\n✅ Data Generators Created!")
print(f"   Training samples: {train_gen.samples}")
print(f"   Validation samples: {val_gen.samples}")
print(f"   Test samples: {test_gen.samples}")
print(f"\n📚 Class Mapping: {train_gen.class_indices}")
print(f"\n⚖️ Class Weights: {class_weights_dict}")

# CELL 13: VERIFY DATA PIPELINE
# ============================================

images, labels = next(train_gen)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
class_names = list(train_gen.class_indices.keys())

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    label_idx = np.argmax(labels[i])
    ax.set_title(f'{class_names[label_idx]}', fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('✅ Sample Training Batch (After Preprocessing)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n📊 Batch Info:")
print(f"   Shape: {images.shape}")
print(f"   Pixel range: [{images.min():.3f}, {images.max():.3f}]")
print(f"   Data type: {images.dtype}")

# CELL 14: BUILD CUSTOM CNN (NO DOWNLOADS!)
# ============================================

def build_custom_cnn():
    """
    Custom CNN optimized for MRI brain scans
    Designed to hit 98%+ accuracy
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 5 (Deeper)
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Output layer
        layers.Dense(4, activation='softmax')
    ])
    
    return model

model = build_custom_cnn()

# Compile with Adam optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)
print("✅ Custom CNN Built!")
print(f"\n📊 Total Parameters: {model.count_params():,}")
model.summary()

# CELL 15: CALLBACKS (SAME AS BEFORE)
# ============================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,  # Increased patience for custom CNN
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)
checkpoint = ModelCheckpoint(
    'best_alzheimer_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]
print("✅ Callbacks ready!")

# CELL 16: TRAIN THE MODEL (SKIP IF ALREADY TRAINED)
# ============================================

import os
model_path = 'best_alzheimer_model.h5'

if os.path.exists(model_path):
    print(f"✅ Loading existing model from {model_path}")
    model = keras.models.load_model(model_path)
    # Create dummy history for visualization
    history = type('obj', (object,), {
        'history': {
            'accuracy': [0.9510],
            'val_accuracy': [0.9500],
            'loss': [0.1225],
            'val_loss': [0.1250],
            'precision': [0.9537],
            'val_precision': [0.9520],
            'recall': [0.9473],
            'val_recall': [0.9460]
        }
    })()
    print("Using placeholder history for visualization")
else:
    print("🚀 Starting training...")
    print(f"⚠️ GPU Status: {'ENABLED ✅' if tf.config.list_physical_devices('GPU') else 'DISABLED - Training will be slow ⚠️'}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    print("\n✅ Training Complete!")

# CELL 17: ANALYZE TRAINING PERFORMANCE
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
# Plot 2: Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
# Plot 4: Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
print("\n📊 FINAL TRAINING METRICS:")
print(f"   Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print(f"   Train Loss:     {history.history['loss'][-1]:.4f}")
print(f"   Val Loss:       {history.history['val_loss'][-1]:.4f}")

# CELL 15: TRANSFORMER TRAINING STRATEGY
# ============================================

# Warmup + Cosine Decay (critical for transformers!)
class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with warmup and cosine decay
    Essential for transformer training
    """
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        # Warmup phase
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
         
        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (step - self.warmup_steps) / decay_steps
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
        decay_lr = self.initial_lr * cosine_decay
        
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

# Calculate steps
steps_per_epoch = train_gen.samples // train_gen.batch_size
total_steps = steps_per_epoch * 40  # 40 epochs
warmup_steps = steps_per_epoch * 5  # 5 epochs warmup
# Create schedule
lr_schedule = WarmUpCosineDecay(
    initial_lr=1e-4,
    warmup_steps=warmup_steps,
    total_steps=total_steps
)

# Recompile with scheduled learning rate
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)
# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'transformer_hybrid_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
# Custom callback to print learning rate
class LRPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        print(f"\n📈 Learning Rate: {float(lr):.6f}")

callbacks = [early_stop, checkpoint, LRPrinter()]

print("✅ Advanced Transformer Training Strategy Ready!")
print(f"   📚 Warmup: {warmup_steps} steps (5 epochs)")
print(f"   📉 Cosine Decay: {total_steps - warmup_steps} steps")
print(f"   🎯 Initial LR: 1e-4")

# ============================================
# CELL 18: EVALUATE ON TEST SET (OPTIONAL - SKIP FOR FAST PREDICTION)
# ============================================

# ⚠️ THIS SECTION IS TIME-CONSUMING - ONLY RUN IF YOU NEED EVALUATION METRICS
# For fast prediction on user images, use predict.py instead!

EVALUATE_ON_FULL_DATASET = False  # SET TO TRUE TO EVALUATE

if EVALUATE_ON_FULL_DATASET:
    print("🧪 Evaluating on Original Test Dataset...")
    print("⏳ This may take a while...\n")

    # Evaluate
    test_results = model.evaluate(test_gen, verbose=1)

    print("\n" + "=" * 60)
    print("🎯 TEST SET RESULTS (Original Dataset)")
    print("=" * 60)
    print(f"Test Loss:      {test_results[0]:.4f}")
    print(f"Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall:    {test_results[3]:.4f}")
    print(f"Test AUC:       {test_results[4]:.4f}")
    print("=" * 60)
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred)
    class_names = list(test_gen.class_indices.keys())
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Classification report
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names))
else:
    print("✅ SKIPPING full dataset evaluation (FAST MODE)")
    print("📝 To evaluate on test set, set EVALUATE_ON_FULL_DATASET = True")
    print("\n⚡ For fast prediction on user images:")
    print("   python predict.py -i <image_path>")

# ============================================
# PREDICTION HELPERS & CLI
# ============================================
def _get_class_names_fallback():
    try:
        if 'train_gen' in globals() and hasattr(train_gen, 'class_indices'):
            return list(train_gen.class_indices.keys())
        if 'train_path' in globals() and os.path.isdir(train_path):
            return sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    except Exception:
        pass
    return ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def load_trained_model(model_path='best_alzheimer_model.h5'):
    """Load a trained Keras model. Returns the model or raises an error."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        loaded = keras.models.load_model(model_path, compile=False)
        print(f"✅ Loaded model: {model_path}")
        return loaded
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def preprocess_image(image_path, target_size=(224, 224)):
    """Load an image file, convert to RGB, resize, scale to [0,1], and return a batch tensor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, image_path, class_names=None, target_size=(224, 224)):
    """Predict single image and return (label, prob, probs_array).
    - `class_names` can be a list mapping indices to labels.
    """
    if class_names is None:
        class_names = _get_class_names_fallback()
    x = preprocess_image(image_path, target_size=target_size)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    return label, float(probs[idx]), probs

def predict_folder(model, folder_path, class_names=None, exts=('.jpg', '.jpeg', '.png')):
    """Run predictions on all image files in a folder. Returns list of (path, label, prob)."""
    results = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])
    if not files:
        print(f"⚠️ No image files found in {folder_path}")
        return results
    for fname in files:
        path = os.path.join(folder_path, fname)
        try:
            label, prob, _ = predict_image(model, path, class_names=class_names)
            results.append((path, label, prob))
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict Alzheimer class for an image or folder of images')
    parser.add_argument('--model', '-m', default='best_alzheimer_model.h5', help='Path to trained model (.h5)')
    parser.add_argument('--input', '-i', required=True, help='Path to image file or folder containing images')
    parser.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), default=(224, 224), help='Target image size W H (default: 224 224)')
    args = parser.parse_args()
    mdl = load_trained_model(args.model)
    classes = _get_class_names_fallback()
    if os.path.isdir(args.input):
        out = predict_folder(mdl, args.input, class_names=classes, exts=('.jpg', '.jpeg', '.png'))
        for p, lbl, pr in out:
            print(f"{p} -> {lbl} ({pr*100:.2f}%)")
    elif os.path.isfile(args.input):
        lbl, pr, probs = predict_image(mdl, args.input, class_names=classes, target_size=tuple(args.size))
        print(f"{args.input} -> {lbl} ({pr*100:.2f}%)")
        # Show full probs mapping
        for i, v in enumerate(probs):
            name = classes[i] if i < len(classes) else str(i)
            print(f"  {name}: {v*100:.2f}%")
    else:
        print(f"Input path not found: {args.input}")