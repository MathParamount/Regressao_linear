import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Regressao:

	def __init__(self,x,y):
		self.coef = None
		self.x = np.array(x)
		self.y = np.array(y)
		self.formula = None
		self.intercept = None

	def fit(self):

		soma_xy = np.sum(self.x* self.y)
		soma_x_quadr = np.sum(self.x * self.x)
		soma_x = np.sum(self.x)
		soma_y = np.sum(self.y)

		tamanho_x = len(self.x)

		media_x = np.mean(self.x)
		media_y = np.mean(self.y)

		#Aplicação da formula: y = ax + b
		
		a = (tamanho_x*soma_xy - soma_x*soma_y) / (tamanho_x * soma_x_quadr - (soma_x ** 2))
		b = media_y - a*media_x

		self.coef = a
		self.intercept = b

		self.formula = lambda x: self.coef * x + self.intercept

		#Gráfico de ajuste
		
		#Lista de cores para destacar outliers
		cores = {
		'b': 'blue',
		'g': 'green'
		}
		
		menor = min(x)
		maior = max(x)
		
		#Criação de lista de cores para ser usada dinamicamente no gráfico
		lista_cores = []

		for i in x:
			if i == menor:
				lista_cores.append(cores['b'])

			elif i == maior:
				lista_cores.append(cores['g'])
			else:
				lista_cores.append('gray')

		plt.scatter(self.x,self.y,marker = 'o', c = lista_cores)
		plt.plot(self.x,self.formula(self.x), label = f"Regressao: y = {a: .2f}x + {b: .2f}",color = "Yellow")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Ajuste de regressao linear")
		plt.legend()
		plt.show()
	
	#Fórmula de previsão do modelo
	def previsao(self,x):
		
		x = np.array(x)
		return self.formula(x)

	#Soma quadrática total (tamanho total do problema)
	def soma_quadratica(self):
		
		media_y = self.y.mean()
		return sum( (self.y - media_y) ** 2)
	
	#Erro quadrático dos dados
	def erro_quadratico(self):
		
		dados = self.previsao(self.x)
		return sum((self.y - dados) ** 2)
	
	#Variação total do modelo
	def soma_regressao_quadratica(self):
		return self.soma_quadratica() - self.erro_quadratico()
	
	def pontuacao(self):
		return 1 - (self.erro_quadratico() / self.soma_quadratica())


#Ajuste do modelo
x = [2,4,6,8,10]
y = [5,10,15,20,30]

if len(x) == len(y):
	modelo = Regressao(x,y)
	modelo.fit()
	
	#Prever valores de x com os fornecidos
	
	previsao = modelo.previsao([10,15,30])
	r2 = modelo.pontuacao()
else:
	print("O tamanho de x eh diferente do de y\n")

print("\n")
print(f"A previsao dos dados são: {previsao}\n")
print(f"A qualidade do modelo é: {r2: .4}\n")
