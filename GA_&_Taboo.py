import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# ======================
# PARÂMETROS GLOBAIS
# ======================
AREA_SIZE = 10   # tamanho do mapa
N_AGV = 7        # numero de carrinhos
N_BASES = 8      # numero de bases
POP_SIZE = 60    # tamanho da população do GA
N_GEN = 100       # numero de gerações do GA
DT = 0.1         # tempo de simulação
MAX_SPEED = 1.0  # vel max carrinho
MIN_SPEED = 0.3  # vel minima carrinho
COLLISION_RADIUS = 0.3  # "raio" de colisão
MIN_DIST = 1.0   # distância mínima entre pontos gerados

# Parâmetros da Busca Tabu
TABU_MAX_ITER = 100
TABU_N_NEIGHBORS = 60
TABU_TENURE = 30

np.random.seed(42)
random.seed(42)

# Contador de avaliações de fitness (simulate)
global_evals = 0

# ======================
# Funções de geração de pontos
# ======================
def generate_points(n_points, area_size, min_dist, existing_points=None, max_tries=1000):
    points = []
    if existing_points is None:
        existing_points = []

    tries = 0
    while len(points) < n_points and tries < max_tries:
        candidate = np.random.rand(2) * area_size
        all_points = points + existing_points
        if all(np.linalg.norm(candidate - p) >= min_dist for p in all_points):
            points.append(candidate)
        tries += 1

    if len(points) < n_points:
        raise RuntimeError(
            f"Não foi possível gerar {n_points} pontos com distância mínima {min_dist} m após {max_tries} tentativas."
        )
    return np.array(points)

# ======================
# GERAÇÃO DO AMBIENTE
# ======================
bases = generate_points(N_BASES, AREA_SIZE, MIN_DIST)
starts = generate_points(N_AGV, AREA_SIZE, MIN_DIST, existing_points=list(bases))

# ======================
# Funções auxiliares
# ======================
def generate_individual():
    """Gera um indivíduo: para cada AGV, (base1, base2, velocidade)."""
    indiv = []
    for i in range(N_AGV):
        pts = random.sample(range(N_BASES), 2)
        vel = random.uniform(MIN_SPEED, MAX_SPEED)
        indiv.append((pts[0], pts[1], vel))
    return indiv

def crossover(parent1, parent2):
    cut = random.randint(1, N_AGV - 1)
    return parent1[:cut] + parent2[cut:]

def mutate(indiv, p=0.2):
    """Mutação "global" do GA (mais forte)."""
    new = []
    for p1, p2, v in indiv:
        if random.random() < p:
            p1, p2 = random.sample(range(N_BASES), 2)
        if random.random() < p:
            v = random.uniform(MIN_SPEED, MAX_SPEED)
        new.append((p1, p2, v))
    return new

def check_collision(pos_t):
    for i in range(len(pos_t)):
        for j in range(i + 1, len(pos_t)):
            if np.linalg.norm(pos_t[i] - pos_t[j]) < COLLISION_RADIUS:
                return True
    return False

def simulate(indiv, return_collision=False):
    """
    Simula AGVs e retorna makespan.
    Se houver colisão:
        - return_collision=False → np.inf
        - return_collision=True → (np.inf, traj, collision_point, collision_frame)
    """
    global global_evals
    global_evals += 1

    positions = starts.copy()
    targets = []
    traj = [positions.copy()]

    for i in range(N_AGV):
        p1, p2, v = indiv[i]
        targets.append([bases[p1], bases[p2], starts[i]])

    t = 0.0
    finished = [False] * N_AGV
    times_finished = [0.0] * N_AGV
    frame = 0

    while not all(finished) and t < 300:
        new_positions = positions.copy()
        for i in range(N_AGV):
            if finished[i]:
                continue
            v = indiv[i][2]
            current_target = targets[i][0]
            direction = current_target - positions[i]
            dist = np.linalg.norm(direction)
            if dist < v * DT:
                new_positions[i] = current_target
                targets[i].pop(0)
                if len(targets[i]) == 0:
                    finished[i] = True
                    times_finished[i] = t
            else:
                new_positions[i] += (direction / dist) * v * DT

        if check_collision(new_positions):
            if return_collision:
                # identifica posição média da primeira colisão
                for a in range(len(new_positions)):
                    for b in range(a + 1, len(new_positions)):
                        if np.linalg.norm(new_positions[a] - new_positions[b]) < COLLISION_RADIUS:
                            collision_point = (new_positions[a] + new_positions[b]) / 2
                            return np.inf, traj, collision_point, frame + 1
            return np.inf

        positions = new_positions
        traj.append(positions.copy())
        t += DT
        frame += 1

    return max(times_finished)

def simulate_trajectory(indiv):
    """Simula trajetória completa (sem colisão) para animação."""
    positions = starts.copy()
    targets = []
    traj = [positions.copy()]
    for i in range(N_AGV):
        p1, p2, v = indiv[i]
        targets.append([bases[p1], bases[p2], starts[i]])

    finished = [False] * N_AGV
    t = 0.0

    while not all(finished) and t < 300:
        new_positions = positions.copy()
        for i in range(N_AGV):
            if finished[i]:
                continue
            v = indiv[i][2]
            current_target = targets[i][0]
            direction = current_target - positions[i]
            dist = np.linalg.norm(direction)
            if dist < v * DT:
                new_positions[i] = current_target
                targets[i].pop(0)
                if len(targets[i]) == 0:
                    finished[i] = True
            else:
                new_positions[i] += (direction / dist) * v * DT
        positions = new_positions
        traj.append(positions.copy())
        t += DT
    return traj

# ======================
# Funções da Busca Tabu
# ======================
def indiv_key(indiv):
    """Chave imutável para guardar no tabu (hash simples do indivíduo)."""
    return tuple(indiv)

def local_mutation(indiv):
    """
    Pequena mudança na solução (vizinho para a Busca Tabu):
    - altera uma base OU
    - inverte a ordem das bases OU
    - ajusta levemente a velocidade.
    """
    new = list(indiv)
    i = random.randrange(N_AGV)
    p1, p2, v = new[i]

    r = random.random()
    if r < 0.4:
        # muda uma das bases
        if random.random() < 0.5:
            p1 = random.randrange(N_BASES)
            while p1 == p2:
                p1 = random.randrange(N_BASES)
        else:
            p2 = random.randrange(N_BASES)
            while p2 == p1:
                p2 = random.randrange(N_BASES)
    elif r < 0.7:
        # troca ordem das bases
        p1, p2 = p2, p1
    else:
        # ajusta velocidade um pouco
        delta = random.uniform(-0.1, 0.1)
        v = max(MIN_SPEED, min(MAX_SPEED, v + delta))

    new[i] = (p1, p2, v)
    return new

def tabu_search(initial_indiv, max_iter=TABU_MAX_ITER,
                n_neighbors=TABU_N_NEIGHBORS, tabu_tenure=TABU_TENURE):
    """Busca Tabu partindo de uma solução inicial."""
    current = initial_indiv
    current_cost = simulate(current)

    best = current
    best_cost = current_cost

    tabu_list = [indiv_key(current)]
    history_best = [best_cost]

    for it in range(max_iter):
        neighbors = []
        for _ in range(n_neighbors):
            neighbor = local_mutation(current)
            key = indiv_key(neighbor)
            if key in tabu_list:
                continue
            cost = simulate(neighbor)
            neighbors.append((neighbor, cost))

        # Filtra apenas vizinhos válidos (sem colisão)
        valid_neighbors = [(ind, c) for ind, c in neighbors if c < np.inf]

        if not valid_neighbors:
            # não há vizinho válido → encerra
            break

        # escolhe o melhor vizinho válido
        neighbor, cost = min(valid_neighbors, key=lambda x: x[1])

        current = neighbor
        current_cost = cost

        # atualiza tabu
        tabu_list.append(indiv_key(current))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # atualiza melhor global
        if current_cost < best_cost:
            best = current
            best_cost = current_cost

        history_best.append(best_cost)

    return best, best_cost, history_best

# ======================
# Algoritmo Genético
# ======================
population = [generate_individual() for _ in range(POP_SIZE)]

history_best_ga = []
history_mean_ga = []

best_overall_ind = None
best_overall_fit = np.inf

global_evals = 0  # zera contador antes do GA
t0_ga = time.perf_counter()

for gen in range(N_GEN):
    fitness = [simulate(ind) for ind in population]

    # Fitness válidos (sem colisão)
    finite_fitness = [f for f in fitness if f < np.inf]

    if finite_fitness:
        best_fit_gen = min(finite_fitness)
        mean_fit_gen = sum(finite_fitness) / len(finite_fitness)
    else:
        best_fit_gen = np.nan
        mean_fit_gen = np.nan

    history_best_ga.append(best_fit_gen)
    history_mean_ga.append(mean_fit_gen)

    # melhor indivíduo da geração (se houver solução válida)
    best_idx_gen = np.argmin(fitness)
    if fitness[best_idx_gen] < np.inf and fitness[best_idx_gen] < best_overall_fit:
        best_overall_fit = fitness[best_idx_gen]
        best_overall_ind = population[best_idx_gen]

    if not np.isnan(best_fit_gen):
        print(f"Geração {gen}: melhor makespan = {best_fit_gen:.2f}")
    else:
        print(f"Geração {gen}: nenhuma solução válida")

    # seleção
    ranked = sorted(zip(population, fitness), key=lambda x: x[1])
    selected = [ind for ind, f in ranked if f < np.inf][:POP_SIZE // 2]

    # Garante pelo menos 2 pais
    if len(selected) < 2:
        needed = 2 - len(selected)
        selected += [generate_individual() for _ in range(needed)]
    while len(selected) < POP_SIZE // 2:
        selected.append(generate_individual())

    # reprodução
    children = []
    while len(children) < POP_SIZE // 2:
        p1, p2 = random.sample(selected, 2)
        child = mutate(crossover(p1, p2))
        children.append(child)

    population = selected + children

t1_ga = time.perf_counter()
evals_ga = global_evals

# ======================
# Pós-processamento com Busca Tabu
# ======================
best_solution_ga = best_overall_ind
best_makespan_ga = best_overall_fit

if best_solution_ga is None or best_makespan_ga == np.inf:
    print("\n[AVISO] GA não encontrou solução válida; não há como iniciar Busca Tabu.")
    use_tabu = False
    best_solution_tabu = None
    best_makespan_tabu = np.inf
    history_best_tabu = []
    evals_tabu = 0
    t_tabu = 0.0
else:
    print(f"\nMelhor solução do GA: makespan = {best_makespan_ga:.2f} s")

    t0_tabu = time.perf_counter()
    best_solution_tabu, best_makespan_tabu, history_best_tabu = tabu_search(best_solution_ga)
    t1_tabu = time.perf_counter()

    evals_tabu = global_evals - evals_ga
    t_tabu = t1_tabu - t0_tabu

    print(f"Melhor solução da Busca Tabu: makespan = {best_makespan_tabu:.2f} s")

    use_tabu = best_makespan_tabu < best_makespan_ga

# ======================
# Relatório de custo/tempo
# ======================
print("\n===== COMPARAÇÃO GA vs BUSCA TABU =====")
print(f"GA   → melhor makespan: {best_makespan_ga:.2f} s, "
      f"tempo: {t1_ga - t0_ga:.3f} s, avaliações: {evals_ga}, "
      f"tempo médio por avaliação: {evals_ga/(t1_ga - t0_ga)}")
if use_tabu:
    print(f"Tabu → melhor makespan: {best_makespan_tabu:.2f} s, "
          f"tempo: {t_tabu:.3f} s, avaliações: {evals_tabu}, "
          f"tempo médio por avaliação: {evals_tabu/t_tabu}")
else:
    print("Tabu → não usada (GA não encontrou solução válida OU Tabu não melhorou).")

# ======================
# Gráfico de convergência GA x Tabu
# ======================
plt.figure()
gens = np.arange(1, N_GEN + 1)
plt.plot(gens, history_best_ga, marker='o', label='GA - melhor por geração')
#plt.plot(gens, history_mean_ga, marker='x', linestyle='--', label='GA - média (válidos)')

if history_best_tabu:
    iters_tabu = np.arange(1, len(history_best_tabu) + 1)
    plt.plot(iters_tabu, history_best_tabu, marker='s', label='Busca Tabu - melhor acumulado')

plt.xlabel('Iteração / Geração')
plt.ylabel('Makespan (s)')
plt.title('Convergência: GA x Busca Tabu')
plt.grid(True)
plt.legend()

# ======================
# Escolha da solução final para animação
# ======================
if use_tabu:
    final_solution = best_solution_tabu
    print("\nUsando solução da Busca Tabu para a animação.")
else:
    final_solution = best_solution_ga
    print("\nUsando solução do GA para a animação.")

# Para a animação, simulamos a trajetória completa (sem colisão)
traj = simulate_trajectory(final_solution)
collision_info = None  # animação agora só mostra solução final escolhida

# ======================
# Animação
# ======================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("Roteamento de AGVs - Melhor Solução (GA/Tabu)")

ax.scatter(bases[:, 0], bases[:, 1], c='red', marker='x', s=100, label='Bases')
ax.scatter(starts[:, 0], starts[:, 1], c='blue', marker='^', s=80, label='Bases AGV')
scat = ax.scatter(starts[:, 0], starts[:, 1], c='blue', label='AGVs')
ax.legend()

def update(frame):
    scat.set_offsets(traj[frame])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=50, blit=True)

# Mostra gráficos + animação
plt.show()
