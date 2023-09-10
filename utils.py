import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face

def underride(d, **options):
    """ добавляет key-value пары в словарь d, если key нет в d """
    for key, val in options.items():
        d.setdefault(key, val)
    return d

def decorate(**options):
    """ Декорирование текущего Axes 
        В функцию передаются именованные аргументы:

        decorate(title='Title',
            xlabel='x',
            ylabel='y')

        Доступные именованные аргументы для Axes здесь:
        https://matplotlib.org/api/axes_api.html

        Так же можно задать цвет фона facecolor для Figure
        Функция также будет включать отображение легенд,
        для предотвращения этого, стоит указать legend=False
    """
    facecolor = options.pop('facecolor', False)
    if facecolor:
        plt.gcf().set(facecolor=facecolor)
    ax = plt.gca()
    if options.pop('legend', True):
        ax.legend()     # если в словаре нет ключа legend, со значением False

    ax.set(**options)
    plt.tight_layout()