import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output
from utils import underride
from math import floor, ceil

class Cell2D:
    """ Базовый класс для 2d клеточных автоматов """
    def __init__(self, n, m=None):
        """ n - число строк, m - число рядов """
        m = n if m is None else m
        self.array = np.zeros((n, m), np.uint8)

    def loop(self, iters=1):
        for i in range(iters):
            self.step()


    def animate(self, frames, size=None, interval=None, iters=1):
        """Animate the automaton.
        
        frames: number of frames to draw
        interval: time between frames in seconds
        iters: number of steps between frames
        """
        # if iters is None:
        #     step = self.step
        # else:
        #     step = self.loop

            
        plt.figure()
        try:
            for i in range(frames-1):
                self.draw(size)
                plt.show()
                if interval:
                    sleep(interval)
                self.loop(iters)
                clear_output(wait=True)
                fig = plt.gcf()
            self.draw(size)
            plt.show()
        except KeyboardInterrupt:
            pass
        # функция вернет 
        return fig

    def draw(self, size=None, **options):
        return draw_array(self.array, size, **options)


def draw_array(array, size=None, **options):
    """ Отрисовка матрицы """
    n, m = array.shape

    # параметры, неопределенные в словаре optopns доопределяются:
    options = underride(options,
                        cmap='Greens',
                        alpha=0.7,
                        vmin=0, vmax=1, 
                        interpolation='none', 
                        origin='upper',
                        extent=[0, m, 0, n])

    plt.axis([0, m, 0, n])
    plt.xticks([])   # убираем
    plt.yticks([])   # деления
    fig = plt.gcf()
    if size is not None:
        fig.set_size_inches(size, size)
    plt.imshow(array, **options)

    return fig

def add_island(array, row=None, col=None, height=1, size=1, noise=None):
    """ приподнимается островок в матрице array на величину height, размера size;
        row & col - координаты центра островка;
        по-умолчанию будет выбран центр окна;
        можно привнести шум, задав noise
     """
    n, m = array.shape
    centerY = n//2 if row is None else row
    centerX = m//2 if col is None else col

    r = size // 2

    # создается островок с размером size с концентрацией val вещества:
    array[centerY-r : centerY+r, centerX-r : centerX+r] += height
    if noise is not None:
        array += noise * np.random.random((n, m)) # вносится шум в матрицу a



def three_frame(world, n_seq):
    """Draw three timesteps.
    
    world: object with step, loop, and draw
    n_seq: 3-tuple, number of steps before each draw
    """
    plt.figure(figsize=(10, 4))

    for i, n in enumerate(n_seq):
        # создается 3 оси в один ряд с индексами 1, 2 и 3
        plt.subplot(1, 3, i+1)
        world.loop(n)
        world.draw()

    plt.tight_layout()
    fig = plt.gcf()
    return fig


def multi_frame(world, n_seq, **options):
    """Draw three timesteps.
    
    world: object with step, loop, and draw
    n_seq: 3-tuple, number of steps before each draw
    """
    m, n = world.array.shape
    options = underride(options,
                    cmap='Greens',
                    alpha=0.7,
                    vmin=0, vmax=1, 
                    interpolation='none', 
                    origin='upper',
                    extent=[0, m, 0, n])
    
    count = len(n_seq)
    rows = ceil(count // 3)

    fig, axs = plt.subplots(rows, 3)
    # fig.set_size_inches
    counter = 0
    for i in range(rows):
        for j in range(3):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if counter < count:
                world.loop(n_seq[counter])
                axs[i, j].imshow(world.array, **options)
                counter += 1

    plt.tight_layout()
    fig = plt.gcf()
    return fig