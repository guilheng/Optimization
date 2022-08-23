# PLANEJAMENTO DO ATENDIMENTO AS ORDENS DE MANUTENÇÃO DA SUPERESTRUTURA FERROVIÁRIA CONSIDERANDO SINCRONISMO, PRECEDÊNCIA E PRIORIDADE
# META-HEURÍSTICA BASED ON ANT COLONY OPTIMIZATION (ACO)

from random import uniform
from copy import deepcopy
import numpy, datetime, math, pandas, sys, os, glob
import matplotlib.pyplot as plt

agora = str(datetime.datetime.now())
file_name = '{} - {}.{} {}h{}m'.format(os.path.basename(sys.argv[0])[:-3], agora[8:10], agora[5:7], agora[11:13], agora[14:16])

with open('{}.txt'.format(file_name), 'w') as arquivo:

	alfa = 1.00
	# Fator do feromônio
	beta = 0.30
	# Fator da distância
	ro = 0.81
	# Taxa de evaporacao

	numiteracoes = 30
	numformigas = 20000

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	def remover_duplicatas(vetor):
		novo_vetor = []
		for i in vetor:
			if i not in novo_vetor:
				novo_vetor.append(i)
		return novo_vetor

	def verificar_loop(loop):
		loop = loop + 1
		return loop

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DECLARAÇÃO DE MATRIZES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	np = 5 # Numero de patios
	nv = 14 # Numero de veiculos maquinas
	na = 24 # Numero de atendimentos
	nr = 6 # Numero de rotas (limite de atendimento por máquina)
	ns = 10 # Numero de servicos

	tt = 480 # Limite de tempo turno do trabalho
	td = 1140 # Tempo de trabalho mais descanso 19h
	tl = 60 # Tempo de setup
	ss = [4, 1, 5, 6, 3, 5, 2, 5, 7, 10, 1, 5, 6, 8, 9, 3, 5, 4, 1, 5, 2, 5, 1, 5] # Solicitacao de Servico demandado pelo trecho/cliente i (i in NT)
	ss = ([-1,] * np) + ss

	tv = [
		[0, 74.85, 156.6, 202.65, 232.65, 28.05, 4.8, 4.8, 4.8, 4.8, 4.8, 23.25, 23.25, 42.3, 42.3, 64.5, 64.5, 64.5, 84.6, 84.6, 93.9, 93.9, 134.1, 113.4, 113.4, 167.55, 167.55, 213.15, 213.15, 0, 74.85, 156.6, 202.65, 232.65],\
		[74.85, 0, 81.75, 127.8, 157.8, 102.9, 79.65, 79.65, 79.65, 79.65, 79.65, 51.6, 51.6, 32.55, 32.55, 10.35, 10.35, 10.35, 9.75, 9.75, 19.05, 19.05, 59.25, 38.55, 38.55, 92.7, 92.7, 138.3, 138.3, 74.85, 0, 81.75, 127.8, 157.8],\
		[156.6, 81.75, 0, 46.05, 76.05, 184.65, 161.4, 161.4, 161.4, 161.4, 161.4, 133.35, 133.35, 114.3, 114.3, 92.1, 92.1, 92.1, 72, 72, 62.7, 62.7, 22.5, 43.2, 43.2, 10.95, 10.95, 56.55, 56.55, 156.6, 81.75, 0, 46.05, 76.05],\
		[202.65, 127.8, 46.05, 0, 30, 230.7, 207.45, 207.45, 207.45, 207.45, 207.45, 179.4, 179.4, 160.35, 160.35, 138.15, 138.15, 138.15, 118.05, 118.05, 108.75, 108.75, 68.55, 89.25, 89.25, 35.1, 35.1, 10.50, 10.50, 202.65, 127.8, 46.05, 0, 30],\
		[232.65, 157.8, 76.05, 30, 0, 260.7, 237.45, 237.45, 237.45, 237.45, 237.45, 209.4, 209.4, 190.35, 190.35, 168.15, 168.15, 168.15, 148.05, 148.05, 138.75, 138.75, 98.55, 119.25, 119.25, 65.1, 65.1, 19.5, 19.5, 232.65, 157.8, 76.05, 30, 0],\
		[28.05, 102.9, 184.65, 230.7, 260.7, 0, 23.25, 23.25, 23.25, 23.25, 23.25, 51.3, 51.3, 70.35, 70.35, 92.55, 92.55, 92.55, 112.65, 112.65, 121.95, 121.95, 162.15, 141.45, 141.45, 195.6, 195.6, 241.2, 241.2, 28.05, 102.9, 184.65, 230.7, 260.7],\
		[15.3, 90.15, 171.9, 217.95, 247.95, 12.75, 0, 0, 0, 0, 0, 38.55, 38.55, 57.6, 57.6, 79.8, 79.8, 79.8, 99.9, 99.9, 109.2, 109.2, 149.4, 128.7, 128.7, 182.85, 182.85, 228.45, 228.45, 15.3, 90.15, 171.9, 217.95, 247.95],\
		[15.3, 90.15, 171.9, 217.95, 247.95, 12.75, 0, 0, 0, 0, 0, 38.55, 38.55, 57.6, 57.6, 79.8, 79.8, 79.8, 99.9, 99.9, 109.2, 109.2, 149.4, 128.7, 128.7, 182.85, 182.85, 228.45, 228.45, 15.3, 90.15, 171.9, 217.95, 247.95],\
		[12, 62.85, 144.6, 190.65, 220.65, 40.05, 0, 0, 0, 0, 0, 11.25, 11.25, 30.3, 30.3, 52.5, 52.5, 52.5, 72.6, 72.6, 81.9, 81.9, 122.1, 101.4, 101.4, 155.55, 155.55, 201.15, 201.15, 12, 62.85, 144.6, 190.65, 220.65],\
		[4.8, 79.65, 161.4, 207.45, 237.45, 23.25, 0, 0, 0, 0, 0, 28.05, 28.05, 47.1, 47.1, 69.3, 69.3, 69.3, 89.4, 89.4, 98.7, 98.7, 138.9, 118.2, 118.2, 172.35, 172.35, 217.95, 217.95, 4.8, 79.65, 161.4, 207.45, 237.45],\
		[4.8, 79.65, 161.4, 207.45, 237.45, 23.25, 0, 0, 0, 0, 0, 28.05, 28.05, 47.1, 47.1, 69.3, 69.3, 69.3, 89.4, 89.4, 98.7, 98.7, 138.9, 118.2, 118.2, 172.35, 172.35, 217.95, 217.95, 4.8, 79.65, 161.4, 207.45, 237.45],\
		[23.25, 51.6, 133.35, 179.4, 209.4, 51.3, 28.05, 28.05, 28.05, 28.05, 28.05, 0, 0, 19.05, 19.05, 41.25, 41.25, 41.25, 61.35, 61.35, 70.65, 70.65, 110.85, 90.15, 90.15, 144.3, 144.3, 189.9, 189.9, 23.25, 51.6, 133.35, 179.4, 209.4],\
		[23.25, 51.6, 133.35, 179.4, 209.4, 51.3, 28.05, 28.05, 28.05, 28.05, 28.05, 0, 0, 19.05, 19.05, 41.25, 41.25, 41.25, 61.35, 61.35, 70.65, 70.65, 110.85, 90.15, 90.15, 144.3, 144.3, 189.9, 189.9, 23.25, 51.6, 133.35, 179.4, 209.4],\
		[42.3, 32.55, 114.3, 160.35, 190.35, 70.35, 47.1, 47.1, 47.1, 47.1, 47.1, 19.05, 19.05, 0, 0, 22.2, 22.2, 22.2, 42.3, 42.3, 51.6, 51.6, 91.8, 81.15, 81.15, 125.25, 125.25, 170.85, 170.85, 42.3, 32.55, 114.3, 160.35, 190.35],\
		[42.3, 32.55, 114.3, 160.35, 190.35, 70.35, 47.1, 47.1, 47.1, 47.1, 47.1, 19.05, 19.05, 0, 0, 22.2, 22.2, 22.2, 42.3, 42.3, 51.6, 51.6, 91.8, 81.15, 81.15, 125.25, 125.25, 170.85, 170.85, 42.3, 32.55, 114.3, 160.35, 190.35],\
		[64.5, 10.35, 92.1, 138.15, 168.15, 92.55, 69.3, 69.3, 69.3, 69.3, 69.3, 41.25, 41.25, 22.2, 22.2, 0, 0, 0, 20.1, 20.1, 29.4, 29.4, 69.6, 48.9, 48.9, 103.05, 103.05, 148.65, 148.65, 64.5, 10.35, 92.1, 138.15, 168.15],\
		[64.5, 10.35, 92.1, 138.15, 168.15, 92.55, 69.3, 69.3, 69.3, 69.3, 69.3, 41.25, 41.25, 22.2, 22.2, 0, 0, 0, 20.1, 20.1, 29.4, 29.4, 69.6, 48.9, 48.9, 103.05, 103.05, 148.65, 148.65, 64.5, 10.35, 92.1, 138.15, 168.15],\
		[52.8, 22.05, 103.8, 149.85, 179.85, 80.85, 57.6, 57.6, 57.6, 57.6, 57.6, 29.55, 29.55, 10.5, 10.5, 0, 0, 0, 31.8, 31.8, 41.1, 41.1, 81.3, 60.6, 60.6, 114.75, 114.75, 160.35, 160.35, 52.8, 22.05, 103.8, 149.85, 179.85],\
		[84.6, 9.75, 72, 118.05, 148.05, 112.65, 89.4, 89.4, 89.4, 89.4, 89.4, 61.35, 61.35, 42.3, 42.3, 20.1, 20.1, 20.1, 0, 0, 9.3, 9.3, 49.5, 28.8, 28.8, 82.95, 82.95, 128.55, 128.55, 84.6, 9.75, 72, 118.05, 148.05],\
		[84.6, 9.75, 72, 118.05, 148.05, 112.65, 89.4, 89.4, 89.4, 89.4, 89.4, 61.35, 61.35, 42.3, 42.3, 20.1, 20.1, 20.1, 0, 0, 9.3, 9.3, 49.5, 28.8, 28.8, 82.95, 82.95, 128.55, 128.55, 84.6, 9.75, 72, 118.05, 148.05],\
		[93.9, 19.05, 62.7, 108.75, 138.75, 121.95, 98.7, 98.7, 98.7, 98.7, 98.7, 70.65, 70.65, 51.6, 51.6, 29.4, 29.4, 29.4, 9.3, 9.3, 0, 0, 40.2, 19.5, 19.5, 73.65, 73.65, 119.25, 119.25, 93.9, 19.05, 62.7, 108.75, 138.75],\
		[93.9, 19.05, 62.7, 108.75, 138.75, 121.95, 98.7, 98.7, 98.7, 98.7, 98.7, 70.65, 70.65, 51.6, 51.6, 29.4, 29.4, 29.4, 9.3, 9.3, 0, 0, 40.2, 19.5, 19.5, 73.65, 73.65, 119.25, 119.25, 93.9, 19.05, 62.7, 108.75, 138.75],\
		[134.1, 59.25, 22.5, 68.55, 98.55, 162.15, 138.9, 138.9, 138.9, 138.9, 138.9, 110.85, 110.85, 91.8, 91.8, 69.6, 69.6, 69.6, 49.5, 49.5, 40.2, 40.2, 0, 20.7, 20.7, 33.45, 33.45, 79.05, 79.05, 134.1, 59.25, 22.5, 68.55, 98.55],\
		[113.4, 38.55, 43.2, 89.25, 119.25, 141.45, 118.2, 118.2, 118.2, 118.2, 118.2, 90.15, 90.15, 81.15, 81.15, 48.9, 48.9, 48.9, 28.8, 28.8, 19.5, 19.5, 20.7, 0, 0, 54.15, 54.15, 99.75, 99.75, 113.4, 38.55, 43.2, 89.25, 119.25],\
		[113.4, 38.55, 43.2, 89.25, 119.25, 141.45, 118.2, 118.2, 118.2, 118.2, 118.2, 90.15, 90.15, 81.15, 81.15, 48.9, 48.9, 48.9, 28.8, 28.8, 19.5, 19.5, 20.7, 0, 0, 54.15, 54.15, 99.75, 99.75, 113.4, 38.55, 43.2, 89.25, 119.25],\
		[167.55, 92.7, 10.95, 35.1, 65.1, 195.6, 172.35, 172.35, 172.35, 172.35, 172.35, 144.3, 144.3, 125.25, 125.25, 103.05, 103.05, 103.05, 82.95, 82.95, 73.65, 73.65, 33.45, 54.15, 54.15, 0, 0, 45.6, 45.6, 167.55, 92.7, 10.95, 35.1, 65.1],\
		[167.55, 92.7, 10.95, 35.1, 65.1, 195.6, 172.35, 172.35, 172.35, 172.35, 172.35, 144.3, 144.3, 125.25, 125.25, 103.05, 103.05, 103.05, 82.95, 82.95, 73.65, 73.65, 33.45, 54.15, 54.15, 0, 0, 45.6, 45.6, 167.55, 92.7, 10.95, 35.1, 65.1],\
		[213.15, 138.3, 56.55, 10.50, 19.5, 241.2, 217.95, 217.95, 217.95, 217.95, 217.95, 189.9, 189.9, 170.85, 170.85, 148.65, 148.65, 148.65, 128.55, 128.55, 119.25, 119.25, 79.05, 99.75, 99.75, 45.6, 45.6, 0, 0, 213.15, 138.3, 56.55, 10.50, 19.5],\
		[213.15, 138.3, 56.55, 10.50, 19.5, 241.2, 217.95, 217.95, 217.95, 217.95, 217.95, 189.9, 189.9, 170.85, 170.85, 148.65, 148.65, 148.65, 128.55, 128.55, 119.25, 119.25, 79.05, 99.75, 99.75, 45.6, 45.6, 0, 0, 213.15, 138.3, 56.55, 10.50, 19.5],\
		[0, 74.85, 156.6, 202.65, 232.65, 28.05, 4.8, 4.8, 4.8, 4.8, 4.8, 23.25, 23.25, 42.3, 42.3, 64.5, 64.5, 64.5, 84.6, 84.6, 93.9, 93.9, 134.1, 113.4, 113.4, 167.55, 167.55, 213.15, 213.15, 0, 74.85, 156.6, 202.65, 232.65],\
		[74.85, 0, 81.75, 127.8, 157.8, 102.9, 79.65, 79.65, 79.65, 79.65, 79.65, 51.6, 51.6, 32.55, 32.55, 10.35, 10.35, 10.35, 9.75, 9.75, 19.05, 19.05, 59.25, 38.55, 38.55, 92.7, 92.7, 138.3, 138.3, 74.85, 0, 81.75, 127.8, 157.8],\
		[156.6, 81.75, 0, 46.05, 76.05, 184.65, 161.4, 161.4, 161.4, 161.4, 161.4, 133.35, 133.35, 114.3, 114.3, 92.1, 92.1, 92.1, 72, 72, 62.7, 62.7, 22.5, 43.2, 43.2, 10.95, 10.95, 56.55, 56.55, 156.6, 81.75, 0, 46.05, 76.05],\
		[202.65, 127.8, 46.05, 0, 30, 230.7, 207.45, 207.45, 207.45, 207.45, 207.45, 179.4, 179.4, 160.35, 160.35, 138.15, 138.15, 138.15, 118.05, 118.05, 108.75, 108.75, 68.55, 89.25, 89.25, 35.1, 35.1, 10.50, 10.50, 202.65, 127.8, 46.05, 0, 30],\
		[232.65, 157.8, 76.05, 30, 0, 260.7, 237.45, 237.45, 237.45, 237.45, 237.45, 209.4, 209.4, 190.35, 190.35, 168.15, 168.15, 168.15, 148.05, 148.05, 138.75, 138.75, 98.55, 119.25, 119.25, 65.1, 65.1, 19.5, 19.5, 232.65, 157.8, 76.05, 30, 0],\
	] # Tempo de viagem de i para j

	sa = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],\
	] # Sincronizacao entre dois atendimento (i in NT e j in NT)

	vs = [
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[1, 0, 0, 0, 0, 1, 0, 0, 0, 0],\
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],\
		[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],\
		[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],\
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],\
	] # Tipo de veiculo que o veiculo v in MD pode executar (veiculo x servico S)

	pd = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	] # Precedencia entre atendimentos (i in NT e j in NT)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DECLARAÇÃO DE VARIÁVEIS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	ot = [0, 0, 0, 0, 0, 42, 112, 112, 244, 300, 300, 135, 135, 222, 222, 167, 167, 195, 222, 222, 240, 240, 42, 112, 112, 150, 150, 209, 209, 0, 0, 0, 0, 0]
	# Tempo de operacao
	P = [1, 10, 10, 1, 100, 100, 1, 1, 1, 1, 10, 10, 1, 1, 1, 100, 100, 1, 10, 10, 1, 1, 10, 10]
	# Prioridade de atendimento

	atendimentos = []
	[atendimentos.append(i) for i in range(na + np)]

	maquinas = []
	[maquinas.append(i) for i in range(nv)]

	lp = [2, 3, 2, 1, 2, 4, 5, 1, 4, 5, 3, 1, 4, 5]
	for i in range(len(lp)):
		lp[i] = lp[i] - 1
	# Pátio de partida

	ck = [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 18]
	# Custo do km por veiculo

	a = [0, 0, 0, 0, 0, 80, 2350, 2350, 3420, 1200, 1200, 3480, 3480, 200, 200, 100, 100, 1300, 4600, 4600, 2300, 2300, 4560, 50, 50, 5800, 5800, 4630, 4630, 0, 0, 0, 0, 0]
	b = [6180, 6180, 6180, 6180, 6180, 80, 2350, 2350, 3420, 1200, 1200, 3480, 3480, 200, 200, 100, 100, 1300, 4600, 4600, 2300, 2300, 4560, 50, 50, 5800, 5800, 4630, 4630, 6180, 6180, 6180, 6180, 6180]
	# Janela de tempo (a-> inicio, b-> final. Valor entre 0 e 480 minutos, 0 A 8 HORAS) (a,b= i in PTP)

	tal = []
	tal_matriz = []
	tal_inicial_atendimento = [0.25,] * np + [0.50,] * na
	for i in range(nv):
		tal.append(deepcopy(tal_matriz))
		for j in range(np + na):
			tal[i].append(deepcopy(tal_inicial_atendimento))
	# Matriz de matrizes de feromônio para atendimentos e pátios

	tal_maq = []
	tal_inicial_maquina = [0.10,] * (nv + 1)
	for i in range(nv + 1):
		tal_maq.append(deepcopy(tal_inicial_maquina))
	# Matriz feromônio para máquinas

	fo_global = 0
	fo = 0
	i = 0
	j = 0
	iteracao = 0

	data_x = [0]
	data_y1 = [0]
	data_y2 = [0]

	prog_i = datetime.datetime.now()
	print(prog_i)

# ===========================================================================================================================================================================================================================
# INÍCIO DO CÓDIGO
# ===========================================================================================================================================================================================================================

	while (iteracao < numiteracoes):
		iteracao = iteracao + 1
		print('__________________________________________________________________________________________')
		print('ITERAÇÃO {} | {}'.format(iteracao, datetime.datetime.now()))
		print('__________________________________________________________________________________________')
		formiga = 0

		vetor_ferom = []
		vetor_sequencia = []
		vetor_rota_sequencia = []
		vetor_maquina_sequencia = []

		while (formiga < numformigas):
			formiga = formiga + 1

			custo_variavel = 0
			prioridade_atendida = [0] * na

			atendimentos_escolhidos = []
			[atendimentos_escolhidos.append([lp[i]]) for i in range(len(lp))]
				
			atendidos = []
			nao_atendidos = atendimentos[:]

			maquinas_usadas = [0]
			maquinas_sequencia = []
			maquinas_disp = maquinas[:]

			assincronos = []
			rota_sincronismo = []

			tempo_rota = [0] * nv
			tempo_chegada_acumulado = [0] * (na + np)
			rota = [0] * nv

			sequencia = [[] for i in range(nv)]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULAR PROBABILIDADE [MÁQUINA]
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

			while (len(maquinas_disp) != 0) and (len(atendidos) != na): # CRITÉRIO DE TÉRMINO DA ROTA DA FORMIGA

				denominador = 0
				somatorio_das_prob = 0
				vetor_prob = [0] * (nv + 2)

				for i in range(len(maquinas_disp)):
					denominador = denominador + (tal_maq[maquinas_usadas[-1]][maquinas_disp[i] + 1])

				for i in range(len(maquinas_disp)):
					vetor_prob[maquinas_disp[i] + 2] = somatorio_das_prob + (tal_maq[maquinas_usadas[-1]][maquinas_disp[i] + 1] / denominador)
					somatorio_das_prob = vetor_prob[maquinas_disp[i] + 2]

				vlr_aleatorio = uniform(0 + 0.00001, 1)

				if ((iteracao > (numiteracoes - 1)) or (iteracao == (numiteracoes / 2)) or (iteracao == 1)) and (formiga == 1):
					print('__________________________________________________________________________________________\nProb máquina: {} >>> {}'.format(vlr_aleatorio, vetor_prob), file=arquivo)

				for i in range(len(vetor_prob) - 1):

					if vetor_prob[i] < vlr_aleatorio <= vetor_prob[i + 1] and vetor_prob[i + 1] != 0:
						maquina = i - 1

				# maquina = int(input('Máquina: '))

				if maquina == -1:
					maquina = int(input('Máquina: '))

				maquinas_disp.remove(maquina)
				maquinas_sequencia.append([maquinas_usadas[-1], maquina + 1])
				maquinas_usadas.append(maquina + 1)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULAR PROBABILIDADE [ATENDIMENTOS]
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

				atendimentos_disp = []

				for i in range(len(nao_atendidos)):

					if (vs[maquina][ss[nao_atendidos[i]] - 1] == 1) or (ss[nao_atendidos[i]] == -1):
						atendimentos_disp.append(nao_atendidos[i])

				loop = 0

				while True:

					loop = verificar_loop(loop)

					if ((len(atendimentos_disp[np:]) == 0) and (ss[atendimentos_escolhidos[maquina][-1]] == -1)) or ((rota[maquina] == nr) and (ss[atendimentos_escolhidos[maquina][-1]] == -1)): # CRITÉRIO DE TÉRMINO DE USO DA MÁQUINA
						cont_patios_finais = 0

						for i in range(1, len(atendimentos_escolhidos[maquina]) + 1):
							
							if ss[atendimentos_escolhidos[maquina][-i]] != -1:
								break
							cont_patios_finais += 1

						vetor_aux1 = atendimentos_escolhidos[maquina][-cont_patios_finais:]
						delta = 0

						for i in range(1, len(vetor_aux1)):

							if vetor_aux1[-i] != vetor_aux1[0]:
								tempo_rota[maquina] = tempo_rota[maquina] - tv[vetor_aux1[-(i - 1)]][vetor_aux1[-i]]
								custo_variavel = custo_variavel - (ck[maquina] * tv[vetor_aux1[-(i - 1)]][vetor_aux1[-i]])
								atendimentos_escolhidos[maquina].pop(-(i - delta))
								delta += 1

						break

					pos_atual = atendimentos_escolhidos[maquina][-1]
					proibidos = []
					vetor_aux1 = []

					denominador = 0
					somatorio_das_prob = 0
					vetor_prob = [0] * ((np + na) + 1)
					vetor_aux2 = numpy.array(tv[pos_atual])
					vetor_aux2 = vetor_aux2[vetor_aux2 != 0]

					try:
						if (ss[atendimentos_escolhidos[maquina][-1]] == -1) and (ss[atendimentos_escolhidos[maquina][-2]] == -1) and (atendimentos_escolhidos[maquina][-1] != atendimentos_escolhidos[maquina][-2]):
							for i in range(np):
								proibidos.append(i)
							proibidos.remove(atendimentos_escolhidos[maquina][-1])
					except:
						pass

					for i in range(len(atendimentos_disp)):

						if (tv[pos_atual][atendimentos_disp[i]] != 0) and (atendimentos_disp[i] not in proibidos):
							denominador = denominador + (((tal[maquina][pos_atual][atendimentos_disp[i]]) ** alfa) * ((1 / tv[pos_atual][atendimentos_disp[i]]) ** beta))

						elif (tv[pos_atual][atendimentos_disp[i]] == 0) and (ss[pos_atual] == -1) and (atendimentos_disp[i] not in proibidos):
							denominador = denominador + (((tal[maquina][pos_atual][pos_atual]) ** alfa) * ((1 / numpy.average(vetor_aux2)) ** beta))

						elif (tv[pos_atual][atendimentos_disp[i]] == 0) and (ss[pos_atual] != -1) and (atendimentos_disp[i] not in proibidos):
							denominador = denominador + (((tal[maquina][pos_atual][atendimentos_disp[i]]) ** alfa) * ((1 / numpy.min(vetor_aux2)) ** beta))

					for i in range(len(atendimentos_disp)):

						if (tv[pos_atual][atendimentos_disp[i]] != 0) and (atendimentos_disp[i] not in proibidos):
							vetor_prob[atendimentos_disp[i] + 1] = somatorio_das_prob + ((((tal[maquina][pos_atual][atendimentos_disp[i]]) ** alfa) * ((1 / tv[pos_atual][atendimentos_disp[i]]) ** beta)) / denominador)
							somatorio_das_prob = vetor_prob[atendimentos_disp[i] + 1]

						elif (tv[pos_atual][atendimentos_disp[i]] == 0) and (ss[pos_atual] == -1) and (atendimentos_disp[i] not in proibidos):
							vetor_prob[atendimentos_disp[i] + 1] = somatorio_das_prob + ((((tal[maquina][pos_atual][pos_atual]) ** alfa) * ((1 / numpy.average(vetor_aux2)) ** beta)) / denominador)
							somatorio_das_prob = vetor_prob[atendimentos_disp[i] + 1]

						elif (tv[pos_atual][atendimentos_disp[i]] == 0) and (ss[pos_atual] != -1) and (atendimentos_disp[i] not in proibidos):
							vetor_prob[atendimentos_disp[i] + 1] = somatorio_das_prob + ((((tal[maquina][pos_atual][atendimentos_disp[i]]) ** alfa) * ((1 / numpy.min(vetor_aux2)) ** beta)) / denominador)
							somatorio_das_prob = vetor_prob[atendimentos_disp[i] + 1]

# ===========================================================================================================================================================================================================================
# ESCOLHA DA ROTA
# ===========================================================================================================================================================================================================================

					vlr_aleatorio = uniform(0, 1)

					for i in range(len(vetor_prob) - 1):

						if vetor_prob[i] < vlr_aleatorio <= vetor_prob[i + 1] and vetor_prob[i + 1] != 0:
							atendimento_atual = i
							erro = []

							# atendimento_atual = int(input('Atendimento Atual: '))

							if (atendimento_atual == -1):
								atendimentos_disp = []
								break

							if (ss[atendimento_atual] != -1):
								atendimentos_disp.remove(atendimento_atual)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AVERIGUAÇÃO DE PRECEDÊNCIA
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

							precedencia_maxima = [-1]
							precedentes = []

							for i in range(na):

								if (pd[i][atendimento_atual - np] == 1) and (atendimento_atual >= np):
									precedentes.append(i + np)

									if ((i + np) in atendidos):
										precedencia_maxima.append(tempo_chegada_acumulado[i + np] + a[i + np] + ot[i + np])
										precedentes.remove(i + np)
										
							precedencia_maxima = max(precedencia_maxima)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. SER PÁTIO
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
							
							if (ss[atendimento_atual] == -1):

								# 1.1 ATENDIMENTO -> PÁTIO
								if (ss[pos_atual] != -1 and ss[atendimento_atual] == -1):

									if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= tt) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
										atendimentos_escolhidos[maquina].append(atendimento_atual)
										tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
										custo_variavel = custo_variavel + (ck[maquina] * tv[pos_atual][atendimento_atual])
										sequencia[maquina].append([pos_atual, atendimento_atual])

								# 1.1 PÁTIO -> PÁTIO
								if (ss[pos_atual] == -1 and ss[atendimento_atual] == -1):

									if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= tt) and (pos_atual != atendimento_atual) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
										atendimentos_escolhidos[maquina].append(atendimento_atual)
										tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
										custo_variavel = custo_variavel + (ck[maquina] * tv[pos_atual][atendimento_atual])
										#sequencia[maquina].append([pos_atual, atendimento_atual])

									elif (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] > tt) and (pos_atual != atendimento_atual) and ((math.floor(tempo_rota[maquina] / td) + 1) * td <= b[atendimento_atual]):
										rota[maquina] = math.floor(tempo_rota[maquina] / td) + 1
										atendimentos_escolhidos[maquina].append(atendimento_atual)
										tempo_rota[maquina] = (td * rota[maquina])
										custo_variavel = custo_variavel + (ck[maquina] * tv[pos_atual][atendimento_atual])
										#sequencia[maquina].append([pos_atual, atendimento_atual])

									elif (pos_atual == atendimento_atual):
										rota[maquina] = math.floor(tempo_rota[maquina] / td) + 1
										atendimentos_escolhidos[maquina].append(atendimento_atual)
										tempo_rota[maquina] = (td * rota[maquina])
										custo_variavel = custo_variavel + (ck[maquina] * tv[pos_atual][atendimento_atual])
										sequencia[maquina].append([pos_atual, atendimento_atual])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. SER ATENDIMENTO
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
							
							if (ss[atendimento_atual] != -1) and (vs[maquina][ss[atendimento_atual] - 1] == 1) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
								tempo_retorno = min(tv[atendimento_atual][:(np - 1)])
								tempo_rota_bkp = tempo_rota[maquina]
								flag = 0

								# 2.1 ATENDIMENTO -> ATENDIMENTO
								if (ss[pos_atual] != -1 and ss[atendimento_atual] != -1):

									if (len(precedentes) == 0) and (precedencia_maxima != -1):

										if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= precedencia_maxima + tl):

											if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

												if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
													tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
													flag = 1

										elif ((precedencia_maxima + tl) % td + ot[atendimento_atual] + tempo_retorno <= tt):

											if (precedencia_maxima + tl >= a[atendimento_atual]) and (precedencia_maxima + tl <= b[atendimento_atual]):
												tempo_rota[maquina] = precedencia_maxima + tl
												rota[maquina] = math.floor(tempo_rota[maquina] / td)
												flag = 1

									elif (len(precedentes) == 0) and (precedencia_maxima == -1):

										if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

												if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
													tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
													flag = 1

								# 2.2 PÁTIO -> ATENDIMENTO
								if (ss[pos_atual] == -1 and ss[atendimento_atual] != -1):

									if (len(precedentes) == 0) and (precedencia_maxima != -1):

										if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= precedencia_maxima + tl):

											if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

												if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
													tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
													flag = 1

												elif (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

													if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

														if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = a[atendimento_atual]
															flag = 1

													elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

														if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
															flag = 1

											elif (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno > tt):

												if ((td * (math.floor(tempo_rota[maquina] / td) + 1)) % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

													if (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
														rota[maquina] = math.floor(tempo_rota[maquina] / td) + 1
														tempo_rota[maquina] = td * rota[maquina] + tv[pos_atual][atendimento_atual]
														flag = 1

													elif (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

														if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

															if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
																rota[maquina] = math.floor(a[atendimento_atual] / td)
																tempo_rota[maquina] = a[atendimento_atual]
																flag = 1

														elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

															if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
																rota[maquina] = math.floor(a[atendimento_atual] / td)
																tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
																flag = 1

										elif (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] < precedencia_maxima + tl):

											if (precedencia_maxima + tl % td + ot[atendimento_atual] + tempo_retorno <= tt):

												if (precedencia_maxima + tl + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (precedencia_maxima + tl <= b[atendimento_atual]):
													rota[maquina] = math.floor(precedencia_maxima + tl / td)
													tempo_rota[maquina] = precedencia_maxima + tl
													flag = 1

												elif (precedencia_maxima + tl + ot[pos_atual] + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

													if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

														if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = a[atendimento_atual]
															flag = 1

													elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

														if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
															flag = 1

											elif ((precedencia_maxima + tl) % td + ot[atendimento_atual] + tempo_retorno > tt):

												if ((td * (math.floor((precedencia_maxima + tl) / td) + 1)) % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

													if (td * (math.floor((precedencia_maxima + tl) / td) + 1) + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (td * (math.floor((precedencia_maxima + tl) / td) + 1) + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
														rota[maquina] = math.floor((precedencia_maxima + tl) / td) + 1
														tempo_rota[maquina] = td * rota[maquina] + tv[pos_atual][atendimento_atual]
														flag = 1

													elif (td * (math.floor((precedencia_maxima + tl) / td) + 1) + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

														if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

															if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
																rota[maquina] = math.floor(a[atendimento_atual] / td)
																tempo_rota[maquina] = a[atendimento_atual]
																flag = 1

														elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

															if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
																rota[maquina] = math.floor(a[atendimento_atual] / td)
																tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
																flag = 1

									elif (len(precedentes) == 0) and (precedencia_maxima == -1):

										if (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

											if (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
												tempo_rota[maquina] = tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual]
												flag = 1

											elif (tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

												if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

													if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
														rota[maquina] = math.floor(a[atendimento_atual] / td)
														tempo_rota[maquina] = a[atendimento_atual]
														flag = 1

												elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

													if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
														rota[maquina] = math.floor(a[atendimento_atual] / td)
														tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
														flag = 1

										elif (tempo_rota[maquina] % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno > tt):

											if ((td * (math.floor(tempo_rota[maquina] / td) + 1)) % td + ot[pos_atual] + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt):

												if (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] >= a[atendimento_atual]) and (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
													rota[maquina] = math.floor(tempo_rota[maquina] / td) + 1
													tempo_rota[maquina] = td * rota[maquina] + tv[pos_atual][atendimento_atual]
													flag = 1

												elif (td * (math.floor(tempo_rota[maquina] / td) + 1) + tv[pos_atual][atendimento_atual] < a[atendimento_atual]):

													if (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= a[atendimento_atual]):

														if (a[atendimento_atual] % td + ot[atendimento_atual] + tempo_retorno <= tt) and (a[atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = a[atendimento_atual]
															flag = 1

													elif (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] > a[atendimento_atual]):

														if (a[atendimento_atual] % td + tv[pos_atual][atendimento_atual] + ot[atendimento_atual] + tempo_retorno <= tt) and (math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual] <= b[atendimento_atual]):
															rota[maquina] = math.floor(a[atendimento_atual] / td)
															tempo_rota[maquina] = math.floor(a[atendimento_atual] / td) * td + tv[pos_atual][atendimento_atual]
															flag = 1

								if flag == 1:
									atendimentos_escolhidos[maquina].append(atendimento_atual)
									atendidos.append(atendimento_atual)
									nao_atendidos.remove(atendimento_atual)
									sequencia[maquina].append([pos_atual, atendimento_atual])
									tempo_chegada_acumulado[atendimento_atual] = tempo_chegada_acumulado[atendimento_atual] + tempo_rota[maquina] - a[atendimento_atual]
									custo_variavel = custo_variavel + (ck[maquina] * tv[pos_atual][atendimento_atual])
									prioridade_atendida[atendimento_atual - np] = P[atendimento_atual - np]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AVERIGUAÇÃO DE SINCRONIZAÇÃO
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

									if sum(sa[atendimento_atual - np]) >= 1:

										for i in range(na):

											if ((sa[atendimento_atual - np][i] == 1) and ((i + np) not in atendidos)):
												assincronos.append(i + np)
												rota_sincronismo.append(rota[maquina])

											if ((sa[atendimento_atual - np][i] == 1) and ((atendimento_atual) in assincronos) and ((i + np) in atendidos)):

												if (tempo_chegada_acumulado[i + np] + a[i + np] >= tempo_chegada_acumulado[atendimento_atual] + a[atendimento_atual]) and ((tempo_chegada_acumulado[i + np] + a[i + np]) % td + ot[atendimento_atual] + tempo_retorno <= tt):
													rota_sincronismo.pop(assincronos.index(atendimento_atual))
													assincronos.remove(atendimento_atual)
													tempo_rota[maquina] = tempo_chegada_acumulado[i + np] + a[i + np]
													rota[maquina] = math.floor(tempo_rota[maquina] / td)
													tempo_chegada_acumulado[atendimento_atual] = tempo_rota[maquina] - a[atendimento_atual]

												else:
													tempo_rota[maquina] = tempo_rota_bkp
													atendimentos_escolhidos[maquina].remove(atendimento_atual)
													atendidos.remove(atendimento_atual)
													nao_atendidos.append(atendimento_atual)
													sequencia[maquina].remove([pos_atual, atendimento_atual])
													tempo_chegada_acumulado[atendimento_atual] = tempo_chegada_acumulado[atendimento_atual] - tempo_rota[maquina] + a[atendimento_atual]
													custo_variavel = custo_variavel - (ck[maquina] * tv[pos_atual][atendimento_atual])
													prioridade_atendida[atendimento_atual - np] = 0

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RESULTADO
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

							else:
								erro.append('Atendimento não é: {} != {} and {} == {} and {} < {} and {} <= {}'.format(ss[atendimento_atual], -1, vs[maquina][ss[atendimento_atual] - 1], 1, len(atendimentos_escolhidos[maquina]), (nr * (rota[maquina] + 1)), tempo_rota[maquina] + ot[pos_atual] + tv[pos_atual][atendimento_atual], b[atendimento_atual]))

							if (((iteracao > (numiteracoes - 1)) or (iteracao == (numiteracoes / 2)) or (iteracao == 1)) and (formiga == 1)) or (loop % 1000 == 0):
								print('__________________________________________________________________________________________', file=arquivo)
								print('ITERAÇÃO {} | FORMIGA {} | LOOP {}'.format(iteracao, formiga, loop), file=arquivo)
								print('Máquina escolhida: {} | Serviço: {}'.format(maquina, vs[maquina]), file=arquivo)
								print('Atendimento escolhido: {} >>> {} | Serviço: {}'.format(pos_atual, atendimento_atual, ss[atendimento_atual]), file=arquivo)
								print('Prob atendimento: {} >>> {}'.format(vlr_aleatorio, vetor_prob), file=arquivo)
								print('\nAtendimentos disponíveis: {}'.format(atendimentos_disp[np:]), file=arquivo)
								print('Atendidos: {} | Não atendidos {}'.format(atendidos, nao_atendidos[np:]), file=arquivo)
								print('Máquinas disponíveis: {}'.format(maquinas_disp), file=arquivo)
								print('Precedentes: {}, {} | Assincronos: {}'.format(precedentes, precedencia_maxima, assincronos), file=arquivo)
								print('Controle de rota / Máquina: {}'.format(rota), file=arquivo)
								print('Tempo da rota (máquinas): {}'.format(tempo_rota), file=arquivo)
								print('Atraso de chegada: {}'.format(tempo_chegada_acumulado), file=arquivo)
								print('Sequência: {}'.format(sequencia), file=arquivo)
								print('Sequência máquinas: {}'.format(maquinas_sequencia), file=arquivo)
								print('\nRota: {}'.format(atendimentos_escolhidos), file=arquivo)
								print('Erro: {}'.format(erro), file=arquivo)

							if (loop % 1000 == 0):
								print('ITERAÇÃO {} | FORMIGA {} | LOOP {}'.format(iteracao, formiga, loop))

							break

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FO E RESULTADOS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

			if (len(assincronos) == 0):
				fo = (100000 * sum(prioridade_atendida)) - (10 * custo_variavel) - (1 * sum(tempo_chegada_acumulado[np:]))
				ferom_depositado = round(((fo / (10 * 100000 * sum(P))) ** 3) * 15, 5)

				for s in range(len(sequencia)):
					sequencia[s] = remover_duplicatas(sequencia[s])

				vetor_ferom.append(ferom_depositado)
				vetor_rota_sequencia.append(sequencia)
				vetor_maquina_sequencia.append(maquinas_sequencia)

				if (fo > fo_global) and (len(assincronos) == 0):
					fo_global = fo
					iteracao_global = iteracao
					formiga_global = formiga
					ferom_depositado_global = ferom_depositado
					atendimentos_escolhidos_globais = atendimentos_escolhidos
					tempo_chegada_global = tempo_rota
					sequencia_global = sequencia
					maquinas_sequencia_global = maquinas_sequencia
					custo_variavel_global = custo_variavel
					tempo_chegada_acumulado_global = tempo_chegada_acumulado
					atendidos_global = atendidos
					assincronos_global = assincronos
					prog_f = datetime.datetime.now() - prog_i
					print('__________________________________________________________________________________________\nRESULTADO FORMIGA {} | ITERAÇÃO {} | FO {} | TEMPO {}\nRota: {}\nTempo da rota: {} | {}\nAtendidos: {} | {}\nTempo de chegada: {} | {}\nCusto variável: {}\nAssincronos: {}\nSequência de máquinas: {}\nSequência: {}\nFO: {}\nFeromônio depositado: {}\n'.format(formiga, iteracao, fo_global, prog_f, atendimentos_escolhidos, tempo_rota, sum(tempo_rota), atendidos, len(atendidos), tempo_chegada_acumulado, sum(tempo_chegada_acumulado), custo_variavel, assincronos, maquinas_sequencia, sequencia, fo, ferom_depositado))

				data_x.append(data_x[-1] + 1)	
				data_y1.append(fo_global)
				data_y2.append(fo)
				
				print('__________________________________________________________________________________________\nRESULTADO FORMIGA {} | ITERAÇÃO {} | FO {} | TEMPO {}\nRota: {}\nTempo da rota: {} | {}\nAtendidos: {} | {}\nTempo de chegada: {} | {}\nCusto variável: {}\nAssincronos: {}\nSequência de máquinas: {}\nSequência: {}\nFO: {}\nFeromônio depositado: {}\n'.format(formiga, iteracao, fo_global, prog_f, atendimentos_escolhidos, tempo_rota, sum(tempo_rota), atendidos, len(atendidos), tempo_chegada_acumulado, sum(tempo_chegada_acumulado), custo_variavel, assincronos, maquinas_sequencia, sequencia, fo, ferom_depositado), file=arquivo)
			
			'''else:
				fo = (100000 * sum(prioridade_atendida)) - (10 * custo_variavel) - (1 * sum(tempo_chegada_acumulado[np:]))

				if fo not in data_y2:
					print('FO: {} | Assincronos: {}'.format(fo, assincronos), file=arquivo)'''

# ===========================================================================================================================================================================================================================
# ATUALIZAÇÃO DO FEROMÔNIO
# ===========================================================================================================================================================================================================================

		for v in range(nv):

			for i in range(np + na):

				for j in range(np + na):
					tal[v][i][j] = ro * tal[v][i][j]

		for f in range(len(vetor_ferom)):

			# ATUALIZAÇÃO DO TAL
			for v in range(nv):

				for i in range(len(vetor_rota_sequencia[f][v])):

					if (vetor_ferom[f] >= 0):
						tal[v][vetor_rota_sequencia[f][v][i][0]][vetor_rota_sequencia[f][v][i][1]] = tal[v][vetor_rota_sequencia[f][v][i][0]][vetor_rota_sequencia[f][v][i][1]] + vetor_ferom[f] / 1			

			# ATUALIZAÇÃO DO TAL GLOBAL
			if fo_global > 0:

				for v in range(nv):

					for i in range(len(sequencia_global[v])):

						if (ferom_depositado_global >= 0):
							tal[v][sequencia_global[v][i][0]][sequencia_global[v][i][1]] = tal[v][sequencia_global[v][i][0]][sequencia_global[v][i][1]] + ferom_depositado_global / 2

			# ATUALIZAÇÃO DO TAL MÁQ
			for i in range(nv + 1):

				for j in range(nv + 1):
					tal_maq[i][j] = ro * tal_maq[i][j]

			for j in range(len(vetor_maquina_sequencia)):

				for i in range(len(vetor_maquina_sequencia[j])):
					tal_maq[vetor_maquina_sequencia[j][i][0]][vetor_maquina_sequencia[j][i][1]] = tal_maq[vetor_maquina_sequencia[j][i][0]][vetor_maquina_sequencia[j][i][1]] + vetor_ferom[f] / 100

			# ATUALIZAÇÃO DO TAL MÁQ GLOBAL
			if fo_global > 0:

				for i in range(len(maquinas_sequencia_global)):
					tal_maq[maquinas_sequencia_global[i][0]][maquinas_sequencia_global[i][1]] = tal_maq[maquinas_sequencia_global[i][0]][maquinas_sequencia_global[i][1]] + ferom_depositado_global / 200

		if (((iteracao > (numiteracoes - 1)) or (iteracao == (numiteracoes / 2)) or (iteracao == 1)) and (formiga == 1)):

			if (len(assincronos) == 0):

				for i in range(np + na):

					for j in range(nv):
						tal_show = [round(i, 5) for i in tal[j][i]]
						print('Tal[{}] {} > n: {}'.format(j, i, tal_show), file=arquivo)

				for i in range(nv + 1):
					tal_show = [round(i, 5) for i in tal_maq[i]]
					print('Tal máquina {} > n: {}'.format(i, tal_show), file=arquivo)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIM DO CÓDIGO
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	for i in range(np + na):

		for j in range(nv):
			tal_show = [round(i, 5) for i in tal[j][i]]
			print('Tal[{}] {} > n: {}'.format(j, i, tal_show), file=arquivo)
			print('Tal[{}] {} > n: {}'.format(j, i, tal_show))

	for i in range(nv + 1):
		tal_show = [round(i, 5) for i in tal_maq[i]]
		print('Tal máquina {} > n: {}'.format(i, tal_show), file=arquivo)
		print('Tal máquina {} > n: {}'.format(i, tal_show))

	res_final = '\nITERAÇÃO GLOBAL {} | FORMIGA GLOBAL {}\nAlfa: {} | Beta: {} | Ro: {} | numiteracoes: {} | numformigas: {} | Tal Inicial: {}\nTempo da rota: {} | {}\nAtendidos: {} | {}\nTempo de chegada: {} | {}\nCusto variável: {}\nAssincronos: {}\nFO: {}\nFeromônio depositado: {}\n'.format(iteracao_global, formiga_global, alfa, beta, ro, numiteracoes, numformigas, tal_inicial_atendimento, tempo_chegada_global, sum(tempo_chegada_global), atendidos_global, len(atendidos_global), tempo_chegada_acumulado_global, sum(tempo_chegada_acumulado_global), custo_variavel_global, assincronos_global, fo_global, ferom_depositado_global)

	print(res_final, file=arquivo)
	print(res_final)

	for i in range(len(atendimentos_escolhidos_globais)):
		print('Máquina {}: {}'.format(i + 1, atendimentos_escolhidos_globais[i]), file=arquivo)
		print('Máquina {}: {}'.format(i + 1, atendimentos_escolhidos_globais[i]))

	fig, ax = plt.subplots()
	ax.plot(data_x, data_y2, label="FO")
	ax.plot(data_x, data_y1, label="FO Global")
	ax.legend()

	print('\n{}\n{}\nTempo de processo: {}'.format(prog_i, datetime.datetime.now(), prog_f), file=arquivo)
	print('\n{}\n{}\nTempo de processo: {}'.format(prog_i, datetime.datetime.now(), prog_f))

	arquivo.close()

	df_file = []
	xls = glob.glob('Resultados.xlsx')
	file = pandas.ExcelFile(xls[0])

	df_file = [file.parse(file.sheet_names[0])]
	df_file.append(pandas.DataFrame({'Documento': [file_name], 'FO': [fo_global], 'Resultado': [res_final]}, columns=['Documento', 'FO', 'Resultado']))

	df = pandas.concat(df_file, ignore_index=True)
	df.to_excel(r'Resultados.xlsx', index=False)

	plt.savefig('{}.png'.format(file_name), bbox_inches='tight')
	plt.show()