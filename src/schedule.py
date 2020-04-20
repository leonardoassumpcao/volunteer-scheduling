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


def data_to_array(table_string=None, names_string=None, str_impossible="X", str_available="O", str_preference="OO", manager_char="#", delimiter="\t"):
	if table_string is None:
		table_string = read_data(default_data_file)
	if names_string is None:
		names_string = read_data(default_volunteers_file)
	names_string = (s for s in names_string.split("\n") if s)
	table_string = (s for s in table_string.split("\n") if s)
	managers = [i for i,s in enumerate(names_string) if s.startswith(manager_char)]

	available  = []  # np.zeros((N, T), dtype=bool)
	preference = []  # np.zeros((N, T), dtype=bool)
	prev_week  = []  # np.empty(N, dtype=int)

	avail_dict = {".": False, "-": False, str_impossible: False, str_available: True, str_preference: True}
	pref_dict = {".": False, "-": False, str_impossible: False, str_available: False, str_preference: True}

	for ss_row in table_string:
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

	return available, preference, prev_week, managers


def show_results(results):
	assert results["Z_array"] is not None, "Problem must be solved before calling show_results(..)"
	print("# APRESENTAÇÃO DA SOLUÇÃO ENCONTRADA")

	solver_stats_dict = results["problem"].solver_stats.__dict__
	print("Solver Stats:\n ", solver_stats_dict)

	alpha, beta, gamma = results["alpha"], results["beta"], results["gamma"]
	penalty_1 = results["penalty_1"].value
	penalty_2 = results["penalty_2"].value
	same_day_bonus = results["same_day_bonus"].value
	print("\n[load_cost]: {}".format(results["load_cost"].value))
	print("\n[penalty_1]: (alpha = {:.2f}) * {:.6g} = {}".format(alpha, (1/alpha) * penalty_1, penalty_1))
	print("\n[penalty_2]: (beta = {:.2f}) * {:.6g} = {}".format(beta,   (1/beta)  *  penalty_2, penalty_2))
	print("\n[same_day_bonus]: (gamma = {:.2f}) * {:.6g} = {}".format(gamma, (1/gamma) * same_day_bonus, same_day_bonus))

	print("\n[PESSOAS POR TURNO]:", results["Z_array"].sum(axis=0), sep="\n", end="\n\n")
	print("[TURNOS POR PESSOA]:", results["Z_array"].sum(axis=1), sep="\n", end="\n\n")

	# rows = ("{:2d} => {}".format(i, row) for (i, row) in enumerate(results["Z_array"], start=1))
	print("[MATRIZ DE ALOCAÇÃO]:")
	print_enumeration(results["Z_array"])


def main(datafile=default_data_file, namesfile=default_volunteers_file, min_staff=5, alpha=1.0, beta=0.7, gamma=0.28, verbose=1, solver_options=None):
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

	* Por enquanto, não estamos usando `preference` e `prev_week`, já que poucos usaram esses recursos na planilha.
	'''
	table_string = read_data(datafile)
	names_string = read_data(namesfile)
	available, preference, prev_week, managers = data_to_array(table_string, names_string)
	N, T = available.shape

	Z = cp.Variable((N, T), name="Z", boolean=True)

	if verbose >= 1:
		print("# Programa de otimização linear com variáveis inteiras (MIP)\n\nDados de Entrada:")
		print("\n[Array: AVAILABLE]", 0+available, sep="\n", end="\n\n")
		print("\n[Array: PREFERENCE]", 0+preference, sep="\n", end="\n\n")

	# load[i] == 'número de turnos alocados para a pessoa i' == sum(Z[i,j] para cada j)
	load = cp.sum(Z, axis=1)

	# Variável para induzir a preferência de soluções sem muita gente contribuindo em um só turno do dia:
	same_day = cp.Variable((N, T//2))

	constraints = [
		# Sempre que A_ij==0, impõe-se Z_ij==0:
		Z <= available,

		# Somatório em i (somatório de cada coluna) deve ser, no mínimo, min_staff:
		cp.sum(Z, axis=0) == min_staff,

		# Restrição para garantir a presença de no mínimo um dos `responsáveis` (managers) a cada turno:
		cp.sum(Z[managers, :], axis=0) >= 1,

		# Forma matricial das desigualdades same_day[:, j//2] <= min(Z[:, j], Z[:, j+1]), com j em (0, 2, 4, ..., T-2):
		same_day <= Z[:, 0::2],
		same_day <= Z[:, 1::2]
	]

	# Expressão para a carga média por pessoa, usando a igualdade cp.sum(load) == cp.sum(Z):
	mean_load = cp.sum(Z) / N

	# Função Objetivo (função a ser MINIMIZADA):
	load_cost = cp.norm_inf(load)  # máximo das entradas do vetor `load`

	# PENALTY_1 DESATIVADO (substituído por zero), vide restrição `Z <= available` acima.
	penalty_1 = cp.Constant(0)  # alpha * cp.sum(cp.pos(Z - available))  # penalizava-se alpha para cada entrada Z[i,j] > available[i,j].
	penalty_2 = beta * cp.sum(cp.scalene(load - mean_load, 1.8, 1.0))  # penalizar cada load[i] longe da média (especialmente acima da média)
	same_day_bonus = gamma * cp.sum(same_day)

	objective = cp.Minimize(load_cost + penalty_1 + penalty_2 - same_day_bonus)   # minimize ('máximo das entradas' do vetor load) + penalidades
	problem = cp.Problem(objective, constraints)   # <= problema de otimização sujeito às restrições acima.

	if verbose >= 1:
		print("# Solucionando problema de otimização.")

	if solver_options is None:
		solver_options = {"verbose": (verbose >= 2)}
	elif "TimeLimit" in solver_options:
		# GAMBIARRA: "cvxpy throws error when Gurobi solver encounters time limit"
		cp.settings.ERROR = [cp.settings.USER_LIMIT]
		cp.settings.SOLUTION_PRESENT = [cp.settings.OPTIMAL, cp.settings.OPTIMAL_INACCURATE, cp.settings.SOLVER_ERROR]
		# CONFERIR: https://github.com/cvxgrp/cvxpy/issues/735

	if verbose >= 2: print("\n", "# # # # # " * 9, "\n", sep="")
	problem.solve(solver=cp.GUROBI, **solver_options)
	if verbose >= 2: print("\n", "# # # # # " * 9, "\n", sep="")

	results_dict = {
		"alpha":       alpha,
		"beta":        beta,
		"gamma":       gamma,
		"Z":           Z,
		"Z_array":     None,
		"objective":   objective,
		"constraints": constraints,
		"problem":     problem,
		"load":        load,
		"load_cost":   load_cost,
		"penalty_1":   penalty_1,
		"penalty_2":   penalty_2,
		"same_day_bonus": same_day_bonus
	}

	if results_dict["Z"] is not None:
		results_dict["Z_array"] = np.asarray(Z.value+0.1, dtype=int)
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


def gamma_tests(min_staff=7, gamma_array=None, fixed_beta=0.7, time_limit=15, verbose=0):
	"""Some tests for different values of gamma."""
	gurobi_options = {"verbose": False, "TimeLimit": time_limit}
	if gamma_array is None:
		gamma_array = np.linspace(0.2, 0.36, 5)

	L = []
	for gamma in gamma_array:
		results = main(min_staff=min_staff, beta=fixed_beta, gamma=gamma, verbose=verbose, solver_options=gurobi_options)
		L.append(results)

		penalty_2 = results["penalty_2"].value
		same_day_bonus = results["same_day_bonus"].value

		print("\n# # # GAMMA = {} # # #".format(gamma))
		print("[load_cost]: {}".format(results["load_cost"].value))
		print("[penalty_2]: (beta = {:.2f}) * {:.6g} = {}".format(fixed_beta, penalty_2/fixed_beta, penalty_2))
		print("[same_day_bonus]: (gamma = {:.2f}) * {:.6g} = {}\n".format(gamma, same_day_bonus/gamma, same_day_bonus))
	return L


if __name__ == "__main__":
	L = gamma_tests(min_staff=7, gamma_array=[0.28])
	result = L.pop()

	names = read_data(default_volunteers_file)
	names = names.split("\n")

	print("[Resultado da Demonstração]:")
	for row, name in zip(result["Z_array"], names):
		row = tuple(("x" if j else " ") for j in row)
		days = zip(row[0::2], row[1::2])  # daily pairs (lunch & dinner)
		days = " | ".join("{} {}".format(day, night) for day, night in days)
		print("[", days, "]:", name)
