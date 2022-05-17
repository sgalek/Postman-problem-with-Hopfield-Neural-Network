import numpy as np
from random import randint
from random import uniform


class Hopfield:
    def __init__(self, cities, d, alpha):
        self.cities = cities
        self.neurons = cities**2
        self.alpha = alpha
        self.distance = d

        self.w = np.zeros([self.neurons, self.neurons])

    def normalizacja(self, x):                  #normalizacja wartości zmiany pojedynczego neuronu
        return 0.5*(1.0+np.tanh(self.alpha*x))

    def training(self, u, A, B, C, D, sigma):   #główna funkcja trenujący
        n = self.cities

        for iteration in range((n**2)):
            x = randint(0, n - 1)
            i = randint(0, n - 1)
            tmpA = 0
            for j in range(n):
                if i != j:
                    tmpA += u[x][j]
            tmpA *= -A
            tmpB = 0
            for y in range(n):
                if x != y:
                    tmpB += u[y][i]
            tmpB *= -B
            tmpC = 0
            for y in range(n):
                for j in range(n):
                    tmpC += u[y][j]
            tmpC -= (n+sigma)
            tmpC *= -C
            tmpD = 0
            for y in range(n):
                if 0 < i < n - 1:
                    tmpD += self.distance[x][y] * (u[y][i + 1] + u[y][i - 1])
                elif i > 0:
                    tmpD += self.distance[x][y] * (u[y][i - 1])
                elif i < n-1:
                    tmpD += self.distance[x][y] * (u[y][i + 1])
            tmpD *= -D
            u[x][i] = self.normalizacja(tmpA + tmpB + tmpC + tmpD)
        return u

    def modifyU(self, A, B, C, D, sigma, iterations):       #Obliczanie macierzy u
        u = np.zeros([self.cities, self.cities])
        for i in range(self.cities):
            for j in range(self.cities):
                u[i][j] = uniform(0, 0.03)

        lastDifference = self.calcDifference(u, A, B, C, D, sigma)
        repeated = 0
        max_repeat = 10                                     #liczba iteracji bez zmiany macierzy u (przerwanie po przekroczeniu)
        for iteration in range(iterations):
            u = self.training(u, A, B, C, D, sigma)
            error = self.calcDifference(u, A, B, C, D, sigma)
            if error == lastDifference:
                repeated += 1
            else:
                repeated = 0

            if repeated > max_repeat:
                break
            lastDifference = error
        return u

    def calcDifference(self, u, A, B, C, D, sigma):     #Obliczenie funkcji celu (E(x))
        tmpA = 0
        n = self.cities
        for x in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        tmpA += u[x][i]*u[x][j]
        tmpA *= (A/2.0)

        tmpB = 0
        for i in range(n):
            for x in range(n):
                for y in range(n):
                    if x != y:
                        tmpB += u[x][i] * u[y][i]
        tmpB *= (B/2.0)

        tmpC = 0
        for x in range(n):
            for i in range(n):
                tmpC += u[x][i]
        tmpC -= 10
        tmpC = tmpC*tmpC
        tmpC *= (C/2.0)

        tmpD = 0
        for x in range(n):
            for y in range(n):
                for i in range(n):
                    if 0 < i < n - 1:
                        tmpD += self.distance[x][y] * u[x][i] * (u[y][i + 1] + u[y][i - 1])
                    elif i > 0:
                        tmpD += self.distance[x][y] * u[x][i] * (u[y][i - 1])
                    elif i < n - 1:
                        tmpD += self.distance[x][y] * u[x][i] * (u[y][i + 1])
        tmpD *= (D/2.0)
        return tmpA+tmpB+tmpC+tmpD