from rss.represent.kan.eff_kan import KAN as EFF_KAN


__all__ = ["EFF_KAN"]


def get_kan(parameter):
    de_para_dict = {'dim_in':2,'dim_hidden':100,'dim_out':1,'num_layers':4}
    for key in de_para_dict.keys():
        param_now = parameter.get(key,de_para_dict.get(key))
        parameter[key] = param_now
    # print('MLP : ',parameter)
    if parameter.get('net_name','SIREN') == "EFF_KAN":
        return EFF_KAN([parameter['dim_in'],*[parameter['dim_hidden']]*parameter['num_layers'],parameter['dim_out']])