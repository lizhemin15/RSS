import torch
import torch.nn as nn


class Fourier_Feature(nn.Module):
    def __init__(self, dim_in, dim_out, feature_type='gaussian', **kwargs):
        super().__init__()
        if dim_out % 2 != 0:
            raise ValueError("The input number is not an even number: "+str(dim_out))
        size = (dim_in, dim_out//2)

        # Check if the provided feature type is valid
        if feature_type not in ['gaussian', 'uniform', 'exponential', 'poisson', 'binomial']:
            raise ValueError('Invalid feature_type provided = ' + feature_type)

        feature_params = {}

        # Initialize the Fourier features based on the selected feature type
        if feature_type == 'gaussian':
            # Gaussian features
            feature_params = {'std': 1, 'mean': 0}
            feature_params.update(kwargs)
            mean = feature_params['mean']
            std = feature_params['std']
            self.B = torch.randn(size) * std + mean
        elif feature_type == 'uniform':
            # Uniform features
            feature_params = {'min_value': 0, 'max_value': 1}
            feature_params.update(kwargs)
            min_value = feature_params['min_value']
            max_value = feature_params['max_value']
            self.B = torch.rand(size) * (max_value - min_value) + min_value
        elif feature_type == 'exponential':
            # Exponential features
            feature_params = {'scale': 1}
            feature_params.update(kwargs)
            scale = feature_params['scale']
            self.B = torch.empty(size).exponential_(1 / scale)
        elif feature_type == 'poisson':
            # Poisson features
            feature_params = {'rate': 1}
            feature_params.update(kwargs)
            rate = feature_params['rate']
            self.B = torch.empty(size).poisson_(rate)
        elif feature_type == 'binomial':
            # Binomial features
            feature_params = {'probabilities': 0.5}
            feature_params.update(kwargs)
            probabilities = feature_params['probabilities']
            self.B = torch.empty(size).bernoulli_(probabilities)
        else:
            raise ValueError('Invalid feature_type provided')

        # Check if the features should be learnable
        if 'learnable' in kwargs and kwargs['learnable']:
            self.B = nn.Parameter(self.B)
        print(kwargs)
        if 'gpu_id' in kwargs and isinstance(kwargs['gpu_id'], int):
            self.B = self.B(kwargs['gpu_id'])            

    def forward(self, x):
        # Perform the forward pass by multiplying input with Fourier features
        # Assuming the shape of x @ self.B is N×m
        x_B_result = x @ self.B
        cos_result = torch.cos(x_B_result)
        sin_result = torch.sin(x_B_result)
        # Concatenate using torch.cat
        concatenated_result = torch.cat((cos_result, sin_result), dim=1)
        return concatenated_result
    

def FeatureMap(parameter):
    de_para_dict = {'dim_in':2,'dim_out':100, 'map_type':'fourier', 'feature_type':'gaussian', 'std':1, 'mean':0, 'gpu_id':None}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    if parameter['map_type'] == 'fourier':
        return Fourier_Feature(**de_para_dict)