from rss.represent.inr import MLP,SIREN


def get_nn(parameter={}):
    net_name = parameter.get('net_name','SIREN')
    if net_name == None:
        net_name = 'None'
    if net_name == 'composition':
        pass
    elif net_name == 'MLP':
        nn = MLP(parameter)
    elif net_name == 'SIREN':
        nn = SIREN(parameter)
    else:
        raise('Wrong net_name = ',net_name)
    return nn