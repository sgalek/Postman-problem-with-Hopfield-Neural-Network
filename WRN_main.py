import numpy as np
from matplotlib import pyplot as plt
from WRN_SiecHopfield import Hopfield

def calc_distance(cities):		#Obliczenie macierzy odległości miast
	n = cities.shape[0]
	d = np.zeros([n, n])
	for i in range(n):
		for j in range(n):
			d[i][j] = np.sqrt(np.square(cities[i][0] - cities[j][0]) + np.square(cities[i][1] - cities[j][1]))
	return d


def set_cities(set):			#Wybór zestawu miast
	city = np.zeros([n, 2])
	if set == 1:
		city[0] = (0.06, 0.70)
		city[1] = (0.08, 0.90)
		city[2] = (0.22, 0.67)
		city[3] = (0.30, 0.20)
		city[4] = (0.35, 0.95)
		city[5] = (0.40, 0.15)
		city[6] = (0.50, 0.75)
		city[7] = (0.62, 0.70)
		city[8] = (0.70, 0.80)
		city[9] = (0.83, 0.20)
	elif set == 2:
		city[0] = (0.25, 0.16)
		city[1] = (0.85, 0.35)
		city[2] = (0.65, 0.24)
		city[3] = (0.70, 0.50)
		city[4] = (0.15, 0.22)
		city[5] = (0.25, 0.78)
		city[6] = (0.40, 0.45)
		city[7] = (0.90, 0.65)
		city[8] = (0.55, 0.90)
		city[9] = (0.60, 0.25)
	return city

n = 10
d = np.zeros([n, n])
city = set_cities(2)
d = calc_distance(city)
hp = Hopfield(n, d, 50.0)
v = hp.modifyU(A=100.0, B=100.0, C=95.0, D=110.0, sigma=1, iterations=1000)
print(v)

path = v				#Przekształcenie danych
x = []
y = []
for i in range(10):
	x.append(city[i][0])
for i in range(10):
	y.append(city[i][1])
						#Konfiguracja okna Pyplot i rysowanie miast
plt.rcParams['toolbar'] = 'None'
plt.figure("Sieć Hopfielda")
plt.scatter(x, y)
plt.grid(True)
plt.axis(False)
for i, j in zip(x, y):
	plt.text(i, j, '({}, {})'.format(i, j))

						#Rysowanie linii
start = []
for col in range(len(path)):
	for row in range(len(path)):
		if path[row][col] == 1:
			if start == []:
				start = row
			else:
				plt.plot([x[start], x[row]], [y[start], y[row]], color=[0, 0.40 + 0.05 * col, 0.0 + 0.08 * col])
				start = row

						#Rysowanie linii powrotnej
for row in range(len(path)):
	if path[row][0] == 1:
		plt.plot([x[start], x[row]], [y[start], y[row]], 'r')

plt.show()
