import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ======================
# PARÂMETROS GLOBAIS
# ======================
AREA_SIZE = 10  #tamanho do mapa
N_AGV = 7       #numero carrinhos
N_BASES = 8     #numero de bases
POP_SIZE = 60   #numero população algoritmos genetico
N_GEN = 30      #numero de gerações
DT = 0.1        #tempo de simualçao
MAX_SPEED = 1.0 #vel max carrinho
MIN_SPEED = 0.3 #vel minima carrinho
COLLISION_RADIUS = 0.3  #espaço que o carrinho ocupa
MIN_DIST = 1.0  # distância mínima entre pontos gerados

np.random.seed(42)
random.seed(42)

# ======================
# Funções de geração de pontos com distância mínima
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
        raise RuntimeError(f"Não foi possível gerar {n_points} pontos com distância mínima {min_dist} m após {max_tries} tentativas.")
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
        for j in range(i+1, len(pos_t)):
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
    positions = starts.copy()
    targets = []
    traj = [positions.copy()]
    for i in range(N_AGV):
        p1, p2, v = indiv[i]
        targets.append([bases[p1], bases[p2], starts[i]])

    t = 0.0
    finished = [False]*N_AGV
    times_finished = [0]*N_AGV
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
                    for b in range(a+1, len(new_positions)):
                        if np.linalg.norm(new_positions[a] - new_positions[b]) < COLLISION_RADIUS:
                            collision_point = (new_positions[a] + new_positions[b]) / 2
                            return np.inf, traj, collision_point, frame+1
            return np.inf

        positions = new_positions
        traj.append(positions.copy())
        t += DT
        frame += 1

    return max(times_finished)

def simulate_trajectory(indiv):
    """Simula trajetória completa (sem colisão) para animação"""
    positions = starts.copy()
    targets = []
    traj = [positions.copy()]
    for i in range(N_AGV):
        p1, p2, v = indiv[i]
        targets.append([bases[p1], bases[p2], starts[i]])

    finished = [False]*N_AGV
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
# Algoritmo Genético
# ======================
population = [generate_individual() for _ in range(POP_SIZE)]

for gen in range(N_GEN):
    fitness = [simulate(ind) for ind in population]
    ranked = sorted(zip(population, fitness), key=lambda x: x[1])
    best_fit = ranked[0][1]
    print(f"Geração {gen}: melhor makespan = {best_fit:.2f}")
    selected = [ind for ind, f in ranked if f < np.inf][:POP_SIZE//2]

    # Garante pelo menos 2 pais para cruzamento
    if len(selected) < 2:
        needed = 2 - len(selected)
        selected += [generate_individual() for _ in range(needed)]
    while len(selected) < POP_SIZE//2:
        selected.append(generate_individual())

    children = []
    while len(children) < POP_SIZE//2:
        p1, p2 = random.sample(selected, 2)
        child = mutate(crossover(p1, p2))
        children.append(child)
    population = selected + children

# ======================
# Seleção da melhor solução (válida ou inválida)
# ======================
fitness_with_collision = [(ind, simulate(ind)) for ind in population]
valid_solutions = [p for p in fitness_with_collision if p[1] != np.inf]

if valid_solutions:
    best_solution = sorted(valid_solutions, key=lambda x: x[1])[0][0]
    collision_info = None
    traj = simulate_trajectory(best_solution)
else:
    # pega melhor inválida e registra colisão
    best_invalid = population[0]
    _, traj, collision_point, collision_frame = simulate(best_invalid, return_collision=True)
    best_solution = best_invalid
    collision_info = (traj, collision_point, collision_frame)

# ======================
# Animação
# ======================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("Roteamento de AGVs - Melhor Solução")

ax.scatter(bases[:,0], bases[:,1], c='red', marker='x', s=100, label='Bases')
ax.scatter(starts[:,0], starts[:,1], c='blue', marker='^', s=80, label='Bases AGV')
scat = ax.scatter(starts[:,0], starts[:,1], c='blue', label='AGVs')
collision_marker = ax.scatter([], [], c='red', marker='s', s=120, label='Colisão', alpha=0.8)
ax.legend()

if collision_info:
    traj, collision_point, collision_frame = collision_info
else:
    collision_point = None
    collision_frame = None

def update(frame):
    scat.set_offsets(traj[frame])
    if collision_info and frame >= collision_frame:
        collision_marker.set_offsets(collision_point)
    return scat, collision_marker

ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=50, blit=True)
plt.show()

#todo: comparar com outra meta heuristica
#avaliar custo/tempo de computação
#Taboo search

