# from vpython import *
from ctypes import pointer
from turtle import color, towards
from wave import Wave_read
from sklearn import neighbors

from sympy import rad
import vpython as vp
import numpy as np

nullVector = vp.vector(0, 0, 0)

def randomVector(a, b):
    """Создается вектор, равномерно распределенными элементами в [a, b)"""
    coords = np.random.uniform(a, b, size=3)
    return vp.vector(*coords)


def limitVector(v):
    """Если magnitude превышает 1, задается значение 1"""
    if v.mag > 1:
        v.mag = 1
    return v

class Boid(vp.cone):
    """Боид представляет собой конус VPython, с атрибутами скорости и осей"""
    def __init__(self, radius=0.03, lenght=0.1):
        position = randomVector(0, 1)
        self.vel = randomVector(0, 1).norm()
        super().__init__(pos=position, radius=radius, lenght=lenght)
        self.axis = lenght * self.vel

    def getNeighbors(self, boids, radius, angle):
        """Возвращает список соседей в видимой области"""
        neighbors = []
        for boid in boids:
            if boid is self:
                continue
            offset = boid.pos - self.pos    # vector

            if offset.mag > radius:
                continue
            # угол между вектором скорости и радиусвектором до другого боида
            diff = self.vel.diff_angle(offset)
            if abs(diff) > angle:
                continue
            neighbors.append(boid)
        return neighbors

    def center(self, boids, radius=1, angle=1):
        """Возвращает вектор, направленный в сторону центра масс боидов,
        находящихся в видимой области
        """
        neighbors = self.getNeighbors(boids, radius, angle)
        vecs = [boid.pos for boid in neighbors]     # радиус-векторы боидов
        return self.vectorTowardCenter(vecs)

    def vectorTowardCenter(self, vecs):
        """Вектор, направленный от self к среднему от векторов vecs"""
        if vecs:
            center = np.mean(vecs)
            toward = vp.vector(center - self.pos)
            return limitVector(toward)
        else:
            return nullVector

    def avoid(self, boids, carrot, radius=0.3, angle=np.pi):
        """Находится вектор центра масс всех объектов в видимой области,
        и возврщается радиус-вектор, направленный в противоположную сторону
        (от центра)"""
        objects = boids + [carrot]      # все объекты сцены
        neighbors = self.getNeighbors(objects, radius, angle)
        vecs = [boid.pos for boid in neighbors]
        return -self.vectorTowardCenter(vecs)

    def align(self, boids, radius=0.5, angle=1):
        
        neighbors = self.getNeighbors(boids, radius, angle)
        vecs = [boid.vel for boid in neighbors]
        return self.vectorTowardCenter(vecs)

    def love(self, carrot):
        """Возвращает вектор, направленный в сторону морковки"""
        toward = carrot.pos - self.pos
        return limitVector(toward)

    def setGoal(self, boids, carrot):
        wAvoid = 10     # весовые 
        wCenter = 3     # коэффициенты
        wAlign = 1      
        wLove = 10      

        self.goal = (
            wCenter * self.center(boids) +
            wAvoid * self.avoid(boids, carrot) + 
            wAlign * self.align(boids) +
            wLove * self.love(carrot)
        )
        self.goal.mag = 1

    def move(self, mu=0.1, dt=0.1):
        self.vel = (1-mu) * self.vel + mu * self.goal
        self.vel.mag = 1
        self.pos += dt * self.vel
        self.axis = self.lenght * self.vel

class World:
    def __init__(self, n=10):
        self.boids = [Boid() for i in range(n)]
        self.carrot = vp.sphere(pos=vp.vector(1,0,0),
                                radius=0.1,
                                color=vp.vector(1,0.5,0))
        self.tracking = False

    def step(self):
        for boid in self.boids:
            boid.setGoal(self.boids, self.carrot)
            boid.move()

        if self.tracking:
            self.carrot.pos = vp.scene.mouse.pos


n = 20
size = 5

world = World(n)
vp.scene.center = world.carrot.pos
vp.scene.autoscale = False

def toggleTracking(world):
    world.tracking = not world.tracking

vp.scene.bind('click', toggleTracking)

while 1:
    vp.rate(10)
    world.step()