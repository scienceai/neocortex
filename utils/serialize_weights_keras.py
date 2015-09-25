from keras.models import Model
import h5py
import json

layer_name_dict = {
    'Dense': 'denseLayer',
    'Dropout': 'dropoutLayer',
    'Embedding': 'embeddingLayer',
    'BatchNormalization': 'batchNormalizationLayer',
    'LSTM': 'rLSTMLayer',
    'GRU': 'rGRULayer',
    'JZS1': 'rJZS1Layer',
    'JZS2': 'rJZS2Layer',
    'JZS3': 'rJZS3Layer'
}

layer_params_dict = {
    'Dense': ['weights', 'activation'],
    'Dropout': ['p'],
    'Embedding': ['weights'],
    'BatchNormalization': ['weights', 'epsilon'],
    'LSTM': ['weights', 'activation', 'inner_activation'],
    'GRU': ['weights', 'activation', 'inner_activation'],
    'JZS1': ['weights', 'activation', 'inner_activation'],
    'JZS2': ['weights', 'activation', 'inner_activation'],
    'JZS3': ['weights', 'activation', 'inner_activation']
}

layer_weights_dict = {
    'Dense': ['W', 'b'],
    'Embedding': ['E'],
    'BatchNormalization': ['gamma', 'beta', 'mean', 'std'],
    'LSTM': ['W_xi', 'W_hi', 'b_i', 'W_xc', 'W_hc', 'b_c', 'W_xf', 'W_hf', 'b_f', 'W_xo', 'W_ho', 'b_o'],
    'GRU': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h'],
    'JZS1': ['W_xz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_hh', 'b_h', 'Pmat'],
    'JZS2': ['W_xz', 'W_hz', 'b_z', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h', 'Pmat'],
    'JZS3': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h']
}

def serialize_from_model(model, weights_filepath):
    if not isinstance(model, Model):
        raise TypeError('must pass in object of type keras.models.Model')

    model_metadata = json.loads(model.to_json())
    weights_file = h5py.File(weights_filepath, 'r')

    layers = []

    for n, layer in enumerate(model_metadata['layers']):
        if layer['name'] == 'Activation':
            layers[n-1]['parameters']['activation'] = layer['activation']
            continue

        layer_params = []

        for param in layer_params_dict[layer['name']]:
            if param == 'weights':
                layer_weights = list(weights_file.get('layer_{}'.format(n)))
                weights = {}
                weight_names = layer_weights_dict[layer['name']]
                for name, w in zip(weight_names, layer_weights):
                    weights[name] = weights_file.get('layer_{}/{}'.format(n, w)).value.tolist()
                layer_params.append(weights)
            else:
                layer_params.append(layer[param])

        layers.append({
            'layerName': layer['name'],
            'parameters': layer_params
        })
