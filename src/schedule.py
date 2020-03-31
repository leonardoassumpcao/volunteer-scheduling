# @author: https://linkedin.com/in/leonardoassumpcao/
# Planilha editável para inserir os horários disponíveis:
# https://docs.google.com/spreadsheets/d/1EGNqISK_-tuaDLxKilAdXIl7Gq1Jc6cMF7Cv1DcI1Wo/edit?usp=sharing

import numpy as np  # Biblioteca mais usada para computação científica. https://numpy.org/
import cvxpy as cp  # Uma biblioteca de otimização matemática! https://www.cvxpy.org/

default_data_file = "../data/tabela.txt"
default_volunteers_file = "../data/voluntarios.txt"


def print_enumeration(iterable, start=1, show_empty_values=False, sep="\n", end="\n"):
	show = lambda j, v: "{:02d}: {}".format(j, v)
	iterable = (str(v) for v in iterable)
	iterable = (show(j, v) for j, v in enumerate(iterable, start=start) if (v or show_empty_values))
	print(*iterable, sep=sep, end=end)


def read_data(fname):
	# http://python-notes.curiousefficiency.org/en/latest/python3/text_file_processing.html
	with open(fname, "r", encoding="utf-8") as ff:
		ss = ff.read()
	return ss


def data_to_array(table_string=None, str_impossible="X", str_available="O", str_preference="OO", delimiter="\t"):
	if table_string is None:
		table_string = read_data(default_data_file)

	available  = []  # np.zeros((N, T), dtype=bool)
	preference = []  # np.zeros((N, T), dtype=bool)
	prev_week  = []  # np.empty(N, dtype=int)

	avail_dict = {".": False, "-": False, str_impossible: False, str_available: True, str_preference: True}
	pref_dict = {".": False, "-": False, str_impossible: False, str_available: False, str_preference: True}

	for i, ss_row in enumerate(table_string.split("\n")):
		if not ss_row:
			continue  # (ignorando linhas vazias)

		ss_row = ss_row.split(delimiter)
		prev_week.append(int(ss_row.pop(0)))

		preference_row, available_row = [], []
		preference.append(preference_row)
		available.append(available_row)

		for j, ss_ij in enumerate(ss_row):
			preference_row.append(pref_dict[ss_ij])
			available_row.append(avail_dict[ss_ij])

	available  = np.asarray(available,  dtype=bool)
	preference = np.asarray(preference, dtype=bool)
	prev_week  = np.asarray(prev_week,  dtype=int)
	# N, T = available.shape

	return available, preference, prev_week


def show_results(results):
	assert results["Z_array"] is not None, "Problem must be solved before calling show_results(..)"
	print("# APRESENTAÇÃO DA SOLUÇÃO ENCONTRADA")

	solver_stats_dict = results["problem"].solver_stats.__dict__
	print("Solver Stats:\n ", solver_stats_dict)

	print("\n[load_cost]: {}".format(results["load_cost"].value))
	print("\n[penalty_1]: {}".format(results["penalty_1"].value))
	print("\n[penalty_2]: {}".format(results["penalty_2"].value))

	print("\n[PESSOAS POR TURNO]:", results["Z_array"].sum(axis=0), sep="\n", end="\n\n")
	print("[TURNOS POR PESSOA]:", results["Z_array"].sum(axis=1), sep="\n", end="\n\n")

	# rows = ("{:2d} => {}".format(i, row) for (i, row) in enumerate(results["Z_array"], start=1))
	print("[MATRIZ DE ALOCAÇÃO]:")
	print_enumeration(results["Z_array"])


def main(datafile=default_data_file, min_staff=5, alpha=0.1, beta=0.01, verbose=1, solver_options=None):
	'''Notação:
	M = min_staff: número mínimo de voluntários por turno.
	i = número (código) da pessoa. vai de 1 a N (talvez 28?)
	j = número (código) do turno.  vai de 1 a T (provavelmente 14)

	A_ij = available[i, j] == True quando pessoa i está disponível no turno j
	P_ij = preference[i, j] == True quando pessoa i tem preferência pelo turno j
	w_i  = prev_week[i] == número de turnos (almoços OU jantas) alocados à pessoa i na semana anterior

	Variável a ser otimizada: Z[i, j] == True quando pessoa i é alocada para colaborar no turno j.

	Variável auxiliar: L_i = load[i] == sum_j Z[i, j] == (soma da linha i de Z) == número de turnos alocados pra pessoa i.

	OBJETIVO: minimizar o máximo dos load[i], conforme disponibilidade das pessoas, orientando-se também pelas preferências.
	'''
	table_string = read_data(datafile)
	available, preference, prev_week = data_to_array(table_string)
	N, T = available.shape

	Z = cp.Variable((N, T), name="Z", boolean=True)

	if verbose >= 1:
		print("# Programa de otimização linear com variáveis inteiras (MIP)\n\nDados de Entrada:")
		print("\n[Array: AVAILABLE]", 0+available, sep="\n", end="\n\n")
		print("\n[Array: PREFERENCE]", 0+preference, sep="\n", end="\n\n")

	# load[i] == 'número de turnos alocados para a pessoa i' == sum(Z[i,j] para cada j)
	load = cp.sum(Z, axis=1)

	constraints = [
		# somatório em i (ou seja, somatório de cada coluna) deve ser, no mínimo, min_staff:
		cp.sum(Z, axis=0) >= min_staff,

		# a restrição abaixo deixaria o problema inviável, pois há poucos dados (e não teria como ter 5 pessoas por turno)
		# Z <= preference,  # (por isso está comentado, enquanto as pessoas ainda não preencheram os dados)
	]
	mean_load = (1 / N) * cp.sum(Z)  # <= usamos a igualdade cp.sum(load) == cp.sum(Z)

	# FUNÇÃO OBJETIVO (função a ser MINIMIZADA): (por enquanto, não estamos usando a matriz `preference`)
	load_cost = cp.norm_inf(load)  # máximo das entradas do vetor `load`

	penalty_1 = alpha * cp.sum(cp.pos(Z - available))  # penalizar 0.01 para cada (i,j) tal que Z[i,j] > available[i,j].
	penalty_2 = beta * cp.norm1(load - mean_load)  # penalizar cada load[i] que estiver longe da média

	objective = cp.Minimize(load_cost + penalty_1 + penalty_2)   # minimize ('máximo das entradas' do vetor load) + penalidades
	problem = cp.Problem(objective, constraints)   # <= problema de otimização sujeito às restrições acima.

	if verbose >= 1:
		print("# Solucionando problema de otimização.")
	if solver_options is None:
		solver_options = {"verbose": (verbose >= 2)}
	elif "TimeLimit" in solver_options:
		# ALERTA GAMBIARRA
		# ASSUNTO: "cvxpy throws error when Gurobi solver encounters time limit"
		cp.settings.ERROR = [cp.settings.USER_LIMIT]
		cp.settings.SOLUTION_PRESENT = [cp.settings.OPTIMAL, cp.settings.OPTIMAL_INACCURATE, cp.settings.SOLVER_ERROR]
		# CONFERIR: https://github.com/cvxgrp/cvxpy/issues/735
	if verbose >= 2: print("\n", "##########" * 10, "\n", sep="")
	problem.solve(solver=cp.GUROBI, **solver_options)
	if verbose >= 2: print("\n", "##########" * 10, "\n", sep="")

	results_dict = {
		"Z":           Z,
		"Z_array":     None,
		"objective":   objective,
		"constraints": constraints,
		"problem":     problem,
		"load":        load,
		"load_cost":   load_cost,
		"penalty_1":   penalty_1,
		"penalty_2":   penalty_2,
	}

	if problem.status == cp.OPTIMAL or results_dict["Z"] is not None:
		results_dict["Z_array"] = np.asarray(Z.value, dtype=int)
		if problem.status != cp.OPTIMAL:
			print("# WARNING. Talvez tenha atingido o tempo limite sem otimalidade?")
			print("Status do problema: {}".format(problem.status))
		if verbose: show_results(results_dict)

	else:
		print("# ERRO: o solver não encontrou uma solução!")
		print("Status do problema: {}".format(problem.status))
		print("# Para as disponibilidades dadas, talvez não seja")
		print("# possível obter `min_staff` pessoas por turno?")

	return results_dict


if __name__ == "__main__":
	print("[Teste simples com M=5]")
	main(min_staff=5, verbose=2)

