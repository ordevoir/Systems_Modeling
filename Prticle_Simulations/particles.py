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


def update_vels_after_collision(r1, r2, v1, v2, distance):
    """Процедурная функция преобразования скоростей при столкновении.
    Модифицирует v1 и v2 in-place.
    
    Для одинаковых частиц в упругом столкновении происходит обмен
    компонентами скоростей вдоль линии центров.
    """
    # Защита от нулевой дистанции
    if distance == 0:
        return
    
    # Вектор нормали (normal vector)
    nx = (r1[0] - r2[0]) / distance
    ny = (r1[1] - r2[1]) / distance
    
    # Проекции скоростей на нормаль (normal velocity components)
    v1n = v1[0] * nx + v1[1] * ny
    v2n = v2[0] * nx + v2[1] * ny
    
    # Обмен нормальных компонент
    delta_vn = v2n - v1n
    
    # Модификация скоростей in-place
    v1[0] += delta_vn * nx
    v1[1] += delta_vn * ny
    v2[0] -= delta_vn * nx
    v2[1] -= delta_vn * ny


def handle_collisions_between_particles(rs, vs, diameter):
    for i in range(len(rs)-1):
        for j in range(i+1, len(rs)):
            r1, r2, v1, v2 = rs[i], rs[j], vs[i], vs[j]
            distance = get_distance(r1, r2)
            if distance <= diameter:
                update_vels_after_collision(r1, r2, v1, v2, distance)
                
                # Разделение перекрывающихся частиц
                overlap = diameter - distance
                eps = 1e-9 if distance < 1e-9 else 0 
                nx = (r1[0] - r2[0]) / (distance + eps)
                ny = (r1[1] - r2[1]) / (distance + eps)
                
                rs[i, 0] += overlap * 0.5 * nx
                rs[i, 1] += overlap * 0.5 * ny
                rs[j, 0] -= overlap * 0.5 * nx
                rs[j, 1] -= overlap * 0.5 * ny


def handle_collisions_with_container(rs, vs, diameter, bounds):
    r = diameter / 2
    xmin, xmax = bounds[0, 0] + r, bounds[1, 0] - r
    ymin, ymax = bounds[0, 1] + r, bounds[1, 1] - r

    # Левая стенка
    mask = np.logical_and(rs[:, 0] < xmin, vs[:, 0] < 0)
    rs[mask, 0] = 2 * xmin - rs[mask, 0]  # Зеркальное отражение
    vs[mask, 0] *= -1

    # Правая стенка
    mask = np.logical_and(rs[:, 0] > xmax, vs[:, 0] > 0)
    rs[mask, 0] = 2 * xmax - rs[mask, 0]
    vs[mask, 0] *= -1
    
    # Нижняя стенка
    mask = np.logical_and(rs[:, 1] < ymin, vs[:, 1] < 0)
    rs[mask, 1] = 2 * ymin - rs[mask, 1]
    vs[mask, 1] *= -1

    # Верхняя стенка
    mask = np.logical_and(rs[:, 1] > ymax, vs[:, 1] > 0)
    rs[mask, 1] = 2 * ymax - rs[mask, 1]
    vs[mask, 1] *= -1


def handle_collisions_with_rect(rs, vs, diameter, rect):
    """
    Идеально упругое отражение кружков от осе-ориентированного прямоугольника (Axis-Aligned Bounding Box - AABB).
    Модифицирует rs и vs на месте.
    
    rs : (N,2) координаты центров
    vs : (N,2) скорости
    diameter : float, диаметр кружков
    rect : [x_min, y_min, width, height]
    """
    r = 0.5 * float(diameter)
    x0, y0, w, h = map(float, rect)
    rect_min = np.array([x0, y0], dtype=float)
    rect_max = np.array([x0 + w, y0 + h], dtype=float)
    eps = 1e-9

    # Ближайшая точка прямоугольника к центру (clamp)
    cp = np.minimum(np.maximum(rs, rect_min), rect_max)
    delta = rs - cp
    dist2 = np.einsum('ij,ij->i', delta, delta)

    # --- 1) Удары снаружи/о ребро/угол: расстояние до AABB <= r и dist>0
    mask_out = (dist2 > 0.0) & (dist2 <= r * r)
    if np.any(mask_out):
        d = np.sqrt(dist2[mask_out])
        n = np.zeros_like(delta[mask_out])
        nz = d > 0.0
        n[nz] = (delta[mask_out][nz].T / d[nz]).T  # нормаль контакта
        # минимальный сдвиг наружу
        rs[mask_out] += (r - d)[:, None] * n
        # идеально упругое отражение: v' = v - 2(v·n)n
        vn = np.einsum('ij,ij->i', vs[mask_out], n)[:, None]
        vs[mask_out] -= 2.0 * vn * n

    # --- 2) Центр внутри (или ровно на границе): выталкиваем к ближайшей стороне и отражаем
    inside = (rs[:, 0] > rect_min[0]) & (rs[:, 0] < rect_max[0]) & \
             (rs[:, 1] > rect_min[1]) & (rs[:, 1] < rect_max[1])
    on_edge = (dist2 == 0.0) & ~inside  # центр ровно на ребре/угле
    mask_in = inside | on_edge
    if np.any(mask_in):
        c = rs[mask_in]
        dl = c[:, 0] - rect_min[0]
        dr = rect_max[0] - c[:, 0]
        db = c[:, 1] - rect_min[1]
        dt = rect_max[1] - c[:, 1]
        dists = np.stack([dl, dr, db, dt], axis=1)

        side = np.argmin(dists, axis=1)  # ближайшая сторона
        normals = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        n = normals[side]

        pen = dists[np.arange(len(side)), side] + r + eps  # чтобы круг оказался СНАРУЖИ
        rs[mask_in] += n * pen[:, None]

        vn = np.einsum('ij,ij->i', vs[mask_in], n)[:, None]
        vs[mask_in] -= 2.0 * vn * n



def one_step_simple(rs, vs, diameter, bounds, walls=None):
    """Один шаг симуляции при наличии квадратной границы bounds"""

    rs += vs
    handle_collisions_with_container(rs, vs, diameter, bounds)
    handle_collisions_between_particles(rs, vs, diameter)

    if walls is not None:
        for wall in walls:
            handle_collisions_with_rect(rs, vs, diameter, wall)



def init_in_grid(   bounds, 
                    n_particles, 
                    density_factor=3,
                    position="center",
                    random_vels=True,
                    ):
    """Инициализирует частицы в квадратной cсетке.
    число частиц может поменяться так, чтобы было квадратом целого числа.
    при random_vels=True компоненты скорости будут случайными в диапазоне (-1., 1.).
    при random_vels=False компоненты скорости будут равны 1.0

    position: доступны значения "center", "left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"

    """
    n_particles = int(np.sqrt(n_particles)) ** 2

    xmin, xmax = bounds[0, 0], bounds[1, 0]
    ymin, ymax = bounds[0, 1], bounds[1, 1]
    xmiddle = (xmin + xmax) / 2
    ymiddle = (ymin + ymax) / 2

    left   = xmiddle - (xmax - xmin) / (2 *density_factor)  # ←
    right  = xmiddle + (xmax - xmin) / (2 *density_factor)  # →
    top    = ymiddle + (xmax - xmin) / (2 *density_factor)  # ↑
    bottom = ymiddle - (xmax - xmin) / (2 *density_factor)  # ↓

    coord_values_x = np.linspace(left, right, int(np.sqrt(n_particles)))
    coord_values_y = np.linspace(bottom, top, int(np.sqrt(n_particles)))
    rs = np.zeros((n_particles, 2), dtype=np.float64)
    counter = 0
    for i in range(int(np.sqrt(n_particles))):
        for j in range(int(np.sqrt(n_particles))):
            rs[counter] = coord_values_x[i], coord_values_y[j]
            counter += 1

    if random_vels:
        vs = np.random.uniform(-1., 1., size=(n_particles, 2))
    else:
        vs = np.ones((n_particles, 2), dtype=np.float64)

    side = (right - left)

    if position == "bottom":
        rs[:, 1] -= bottom - ymin - side / 2
    if position == "top":
        rs[:, 1] += bottom - ymin - side / 2
    if position == "left":
        rs[:, 0] -= left - xmin - side / 2
    if position == "right":
        rs[:, 0] += left - xmin - side / 2
        
    if position == "bottom-left":
        rs[:, 0] -= left - xmin - side / 2
        rs[:, 1] -= bottom - ymin - side / 2
    if position == "top-left":
        rs[:, 0] -= left - xmin - side / 2
        rs[:, 1] += bottom - ymin - side / 2
    if position == "top-right":
        rs[:, 0] += left - xmin - side / 2
        rs[:, 1] += bottom - ymin - side / 2
    if position == "bottom-right":
        rs[:, 0] += left - xmin - side / 2
        rs[:, 1] -= bottom - ymin - side / 2
    
    return rs, vs

