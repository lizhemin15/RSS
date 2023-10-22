#from rss.tasks.NeRF import nerf
from rss.tasks.rssnet import rssnet

__all__ = ['rssnet']


def get_task(parameters):
    """
    Get the task class.

    Args:
        task_name (str): The name of the task. Options are:
            - nerf: The task for NeRF.


    Returns:
        task (class): The task class.
    Raises:
        ValueError: If the input task name is not one of the options.
    """
    task_p = parameters.get('task_p',{'task_name':'completion'})
    task_name = task_p.get('task_name','completion')
    if task_name == 'nerf':
        # TODO : add nerf
        return nerf
    elif task_name in ['completion','denoising']:
        return rssnet(parameters)
    else:
        raise ValueError('Task [%s] not recognized.' % task_name)



