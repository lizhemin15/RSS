""" rss
"""
__title__ = 'rss'
__version__ = '0.0.1'
#__build__ = 0x021300
__author__ = 'Zhemin Li'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023 Zhemin Li'


## Top Level Modules

# from rss.represent import get_nn

from rss.tasks import rssnet
from rss.represent import get_reg,get_nn


__all__ = ['rssnet','get_reg','get_nn']


        



