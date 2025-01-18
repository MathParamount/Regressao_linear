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

<div align="center">
<img src="https://github.com/user-attachments/assets/44597635-2ec3-4085-a7cf-8c54c900622d" width="400px" />
</div>

### Gráfico

Nessa parte do código está sendo implementado a parte de visualização do gráfico em que foi feito um enfoque em tornar o gráfico o mais legível, pois foi criado uma separação de cores dos menores para maiores valores. 
```
cores = {
		'b': 'blue',
		'g': 'green'
		}
		
		menor = min(x)
		maior = max(x)
		
		lista_cores = []

		for i in x:
			if i == menor:
				lista_cores.append(cores['b'])

			elif i == maior:
				lista_cores.append(cores['g'])
			else:
				lista_cores.append('gray')
```

Usamos as chamadas padrões da biblioteca matplotlib.pyplot e fizemos a estrutura do gráfico.
```
plt.scatter(self.x,self.y,marker = 'o', c = lista_cores)
		plt.plot(self.x,self.formula(self.x), label = f"Regressao: y = {a: .2f}x + {b: .2f}",color = "Yellow")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Ajuste de regressao linear")
		plt.legend()
		plt.show()
```

### Funções suporte
Foi criado funções que manipulam os dados de entrada e realizam cálculos com base na função linear principal. Isso foi feito para a definição das fórmulas de previsão, soma quadrática e erro_quadrático. 	Essas funções são fundamentais para o modelo de regressão linear.

- Função para previsão:
Isso usa cada valor de x como entrada no parâmetro da função e é chamado a self.formula que é a função linear que realiza os cálculos de regressão do programa.
  
- Função para soma quadrática:
Retorna o erro quadrático com base nos valores da média dos pontos. Assim quanto maior essa medida pior o modelo, pois maior a variabilidade dos dados que não está sendo explicado pelo modelo. Também como está sendo elevado ao quadrado o modelo é penalizado por erros maiores.
  
- Função para erro quadrático:
  Indica que em média os valores previstos são diferentes dos valores reais. Quanto menor erro quadrático mais ajustado está os dados ao modelo.

Também foi criado duais funções para medir a soma da regressão quadrática e a pontuação que mede o quão bem o modelo está do esperado, pois se a pontuação está mais próximo de 1 mais coerente é o sistema.

```
	def previsao(self,x):
		
		x = np.array(x)
		return self.formula(x)

	def soma_quadratica(self):
		
		media_y = self.y.mean()
		return sum( (self.y - media_y) ** 2)
	
	def erro_quadratico(self):
		
		dados = self.previsao(self.x)
		return sum((self.y - dados) ** 2)
	
	def soma_regressao_quadratica(self):
		return self.soma_quadratica() - self.erro_quadratico()
	
	def pontuacao(self):
		return 1 - (self.erro_quadratico() / self.soma_quadratica())
```

Por fim, é feito os ajustes com os dados de entrada para serem manipulados ao longo do código e exibir o resultado para previsão dos dados e a qualidade do modelo.

```
x = [2,4,6,8,10]
y = [5,10,15,20,30]

if len(x) == len(y):
	modelo = Regressao(x,y)
	modelo.fit()

	previsao = modelo.previsao([10,15,30])
	r2 = modelo.pontuacao()
else:
	print("O tamanho de x eh diferente do de y\n")

print("\n")
print(f"A previsao dos dados são: {previsao}\n")
print(f"A qualidade do modelo é: {r2: .4}\n")
```

Uma amostra para o resultado gerado a partir do código:

<div align="center">
<img src="https://github.com/user-attachments/assets/a6dedc0e-768b-4f31-b0ba-e08e72c42d6e" width="400px" />
</div>
