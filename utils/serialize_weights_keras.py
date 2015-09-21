from keras.models import Model
import json

def serialize_from_model(model):
    if not isinstance(model, Model):
        raise TypeError('must pass in object of type keras.models.Model')

    model_metadata = json.loads(model.to_json())
