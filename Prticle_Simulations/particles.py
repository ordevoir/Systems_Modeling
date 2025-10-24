import numpy as np


def get_distance(r1, r2):
    return np.sqrt(np.sum((r1 - r2)**2))


def get_transform_pair_for_equal_prticles(r1, r2, distance):
    """Возвращает матрицы перобразования для пары одинаковых частиц"""
    # защита от нулевой дистанции
    if distance == 0:
        n = np.array([1.0, 0.0])
    else:
        n = (r1 - r2) / distance

    cos, sin = n[0], n[1]
    
    transform = np.array([
        [cos,  sin],
        [-sin, cos],
    ])
    reverse = np.array([
        [cos, -sin],
        [sin,  cos],
    ])
    return transform, reverse


def vels_after_collision(r1, r2, v1, v2, distance):
    """Функция, производящая преобразование скоростей при столкновении
    """
    transform, reverse = get_transform_pair_for_equal_prticles(r1, r2, distance)

    # Переход в собственную СК
    v1_trans = np.dot(transform, v1)
    v2_trans = np.dot(transform, v2)

    # Обмен скоростями
    v1_trans[0], v2_trans[0] = v2_trans[0], v1_trans[0]
    
    # переход обратно в глобальную СК
    v1_reverse = np.dot(reverse, v1_trans)
    v2_reverse = np.dot(reverse, v2_trans)

    return v1_reverse, v2_reverse


def one_step_simple(rs, vs, diameter, bounds):
    """Один шаг симуляции при наличии квадратной границы bounds"""

    # перемещение
    rs += vs

    # столкновение с внешними стенками
    mask = np.logical_or(rs<=bounds[0]+diameter//2, rs>=bounds[1]-diameter//2)
    rs[mask] -= vs[mask]
    vs[mask] = -vs[mask]
    
    # соударение шариков между собой
    for i in range(len(rs)-1):
        for j in range(i+1, len(rs)):
            r1, r2, v1, v2 = rs[i], rs[j], vs[i], vs[j]
            distance = get_distance(r1, r2)
            if distance <= diameter:
                vs[i], vs[j] = vels_after_collision(r1, r2, v1, v2, distance) 
                rs[i] += vs[i]
                rs[j] += vs[j]


def init_in_center_grid(bounds, 
                   n_particles, 
                   density_factor=6,
                   random_vels=True,
                   ):
    """Инициализирует частицы по центру в квадратной cсетке.
    число частиц может поменяться так, чтобы было квадратом целого числа.
    при random_vels=True компоненты скорости будут случайными в диапазоне (-1., 1.).
    при random_vels=False компоненты скорости будут равны 1.0
    """
    n_particles = int(np.sqrt(n_particles)) ** 2

    left = bounds[1] // 2 - bounds[1] // density_factor
    right = bounds[1] - left
    
    coord_values = np.linspace(left, right, int(np.sqrt(n_particles)))
    rs = np.zeros((n_particles, 2), dtype=np.float16)
    counter = 0
    for i in range(int(np.sqrt(n_particles))):
        for j in range(int(np.sqrt(n_particles))):
            rs[counter] = coord_values[i], coord_values[j]
            counter += 1

    if random_vels:
        vs = np.random.uniform(-1., 1., size=(n_particles, 2))
    else:
        vs = np.ones((n_particles, 2), dtype=np.float64)
    return rs, vs