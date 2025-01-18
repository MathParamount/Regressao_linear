# Modelo de regressão linear

## Definição
Este projeto utiliza um modelo de regressão linear para prever valores de uma variável com base em dados prévios. Inclui cálculos estatísticos (como erro absoluto médio, soma dos quadrados residuais e variação total) e gera um gráfico que combina a linha de regressão com os pontos fornecidos como entrada.
Pode ser aplicado em contextos como previsão de falhas em equipamentos e estimativa de prazos para tarefas.

## Objetivo
Esse trabalho surgiu do interesse do autor em fazer uma aplicação prática com a regressão linear, pois isso é o básico para entendimento de machine learning, em que na maioria das vezes tenta prever os dados usando como base as informações atuais.

## Instalações

É fundamental instalar as bibliotecas que será usadas no código. Assim, vá para o terminal do computador e digite:
```
pip install numpy
pip install pandas
pip install matplotlib
```

No início do código importe as bibliotecas a serem usadas e faça as nomeações intuitivas.
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Código
Primeiramente é criado uma classe chamada "Regressão" que vai servir como um escopo para as parte da inicializações das variáveis e funções.
```
class Regressao:

	def __init__(self,x,y):
		self.coef = None
		self.x = np.array(x)
		self.y = np.array(y)
		self.formula = None
		self.intercept = None
```

Logo em diante foi criado uma função para a criação das operações do x, y e da função de regressão linear que será chamado e feito os cálculos automaticamente na parte do ajuste final com os dados de input. 
```
def fit(self):

		soma_xy = np.sum(self.x* self.y)
		soma_x_quadr = np.sum(self.x * self.x)
		soma_x = np.sum(self.x)
		soma_y = np.sum(self.y)

		tamanho_x = len(self.x)

		media_x = np.mean(self.x)
		media_y = np.mean(self.y)
```

Em seguida é feito a criação do coeficiente angular e linear da função linear de regressão. 
```
		a = (tamanho_x*soma_xy - soma_x*soma_y) / (tamanho_x * soma_x_quadr - (soma_x ** 2))
		b = media_y - a*media_x

		self.coef = a
		self.intercept = b

		self.formula = lambda x: self.coef * x + self.intercept
```

Afinal, matematicamente o coeficiente angular("slope") é a inclinação da reta do gráfico, que coincide com a tangente do ângulo formado e o eixo, assim, isso pode ser usado para definir qualquer ponto da função com base no valor do "slope" multiplicado por um valor unitário, que normalmente é chamado de x.


## Hipóteses


## Conclusão
