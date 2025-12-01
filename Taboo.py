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
DT = 0.1         # tempo de simulação
MAX_SPEED = 1.0  # vel max carrinho
MIN_SPEED = 0.3  # vel minima carrinho
COLLISION_RADIUS = 0.3  # "raio" de colisão
MIN_DIST = 1.0   # distância mínima entre pontos gerados

# Parâmetros da Busca Tabu
TABU_MAX_ITER = 200          # número máximo de iterações da Tabu
TABU_N_NEIGHBORS = 60      # quantos vizinhos são gerados por iteração
TABU_TENURE = 30           # tamanho da lista tabu (memória)

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
    """
    Gera um indivíduo: para cada AGV, (base1, base2, velocidade).
    """
    indiv = []
    for i in range(N_AGV):
        pts = random.sample(range(N_BASES), 2)
        vel = random.uniform(MIN_SPEED, MAX_SPEED)
        indiv.append((pts[0], pts[1], vel))
    return indiv

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
    """Busca Tabu partindo de uma solução inicial aleatória."""
    current = initial_indiv
    current_cost = simulate(current)

    best = current
    best_cost = current_cost

    tabu_list = [indiv_key(current)]

    history_best = [best_cost]
    history_current = [current_cost]

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
            print(f"Iteração {it}: nenhum vizinho válido, encerrando Busca Tabu.")
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

        history_current.append(current_cost)
        history_best.append(best_cost)

        print(f"Iteração {it+1}: custo atual = {current_cost:.2f}, "
              f"melhor até agora = {best_cost:.2f}")

    return best, best_cost, history_best, history_current

# ======================
# RODANDO APENAS BUSCA TABU
# ======================
global_evals = 0
initial_solution = generate_individual()

t0 = time.perf_counter()
best_solution, best_cost, history_best, history_current = tabu_search(initial_solution)
t1 = time.perf_counter()

total_time = t1 - t0
total_evals = global_evals

print("\n===== RESULTADOS BUSCA TABU (SEM GA) =====")
print(f"Melhor makespan: {best_cost:.2f} s")
print(f"Tempo de computação: {total_time:.3f} s")
print(f"Número de avaliações (simulate): {total_evals}")

# ======================
# Gráfico de convergência da Tabu
# ======================
iters = np.arange(len(history_best))

plt.figure()
plt.plot(iters, history_current, marker='x', linestyle='--', label='Custo solução atual')
plt.plot(iters, history_best, marker='o', label='Melhor custo acumulado')
plt.xlabel('Iteração da Busca Tabu')
plt.ylabel('Makespan (s)')
plt.title('Convergência da Busca Tabu')
plt.grid(True)
plt.legend()

# ======================
# Animação da melhor solução
# ======================
traj = simulate_trajectory(best_solution)

fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("Roteamento de AGVs - Melhor Solução (Busca Tabu)")

ax.scatter(bases[:, 0], bases[:, 1], c='red', marker='x', s=100, label='Bases')
ax.scatter(starts[:, 0], starts[:, 1], c='blue', marker='^', s=80, label='Bases AGV')
scat = ax.scatter(starts[:, 0], starts[:, 1], c='blue', label='AGVs')
ax.legend()

def update(frame):
    scat.set_offsets(traj[frame])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=50, blit=True)

plt.show()
