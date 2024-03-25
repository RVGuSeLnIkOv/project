from keras.models import load_model

def load_infection_model():
    model_path = "model/infection_segmentation.keras"
    loaded_model = load_model(model_path)
    return loaded_model

def load_lungs_model():
    model_path = "model/lungs_segmentation.keras"
    loaded_model = load_model(model_path)
    return loaded_model