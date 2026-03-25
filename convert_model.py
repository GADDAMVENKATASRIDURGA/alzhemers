import tensorflow as tf
from keras.models import load_model
from keras.layers import InputLayer

# custom InputLayer loader
def custom_input_layer(**config):
    config.pop("batch_shape", None)
    config.pop("optional", None)
    return InputLayer(**config)

model = load_model(
    "best_alzheimer_model.h5",
    compile=False,
    custom_objects={"InputLayer": custom_input_layer}
)

model.save("best_alzheimer_model.keras")

print("Model converted successfully!")