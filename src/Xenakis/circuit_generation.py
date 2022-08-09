import numpy as np
import pennylane as qml
import random


def run_vqe(cost_fn, initial_theta, max_iterations=150, conv_tol=1e-9):
    energy = 1e9
    theta = qml.numpy.array(initial_theta, requires_grad=True)
    opt = qml.AdamOptimizer()
    for i in range(max_iterations):
        prev_energy = energy
        theta, energy = opt.step_and_cost(cost_fn, theta)
        if np.abs(energy - prev_energy) < conv_tol:
            break
    return energy, theta


def string_to_gate(string, n_qubits):
    # first six bits of string define gate time, rest define which qubits to apply to
    n_params = 0
    complexity = 0  # ad-hoc value to assess complexity of gate
    gate_string = string[:6]
    qubit_string = string[6:]
    qubit_seed = int(qubit_string, 2)
    qubits = np.random.RandomState(seed=qubit_seed).permutation(n_qubits)
    if gate_string == '000000':
        def gate(theta):
            qml.RX(theta, wires=qubits[0])

        n_params = 1
        complexity = 2
    elif gate_string == '000001':
        def gate(theta):
            qml.RY(theta, wires=qubits[0])

        n_params = 1
        complexity = 2
    elif gate_string == '000010':
        def gate(theta):
            qml.RZ(theta, wires=qubits[0])

        n_params = 1
        complexity = 2
    elif gate_string == '000100' and n_qubits > 1:
        def gate():
            qml.CY(wires=qubits[:2])

        complexity = 2
    elif gate_string == '000101' and n_qubits > 1:
        def gate():
            qml.CZ(wires=qubits[:2])

        complexity = 2
    elif gate_string == '000110' and n_qubits > 1:
        def gate(theta):
            qml.CRX(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '000111' and n_qubits > 1:
        def gate(theta):
            qml.CRY(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '001000' and n_qubits > 1:
        def gate(theta):
            qml.CRZ(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '001001':
        def gate():
            qml.PauliX(wires=qubits[0])
    elif gate_string == '001010':
        def gate():
            qml.PauliY(wires=qubits[0])
    elif gate_string == '001011':
        def gate():
            qml.PauliZ(wires=qubits[0])
    elif gate_string == '001100':
        def gate(theta):
            qml.PhaseShift(theta, wires=qubits[0])

        n_params = 1
        complexity = 1
    elif gate_string == '001101' and n_qubits > 3:
        def gate():
            qml.QubitCarry(wires=qubits[:4])

        complexity = 3
    elif gate_string == '001111' and n_qubits > 2:
        def gate():
            qml.QubitSum(wires=qubits[:3])

        complexity = 3
    elif gate_string == '010000':
        def gate(theta, phi, omega):
            qml.Rot(theta, phi, omega, wires=qubits[0])

        n_params = 3
        complexity = 2
    elif gate_string == '010001':
        def gate():
            qml.S(wires=qubits[0])
    elif gate_string == '010010' and n_qubits > 1:
        def gate():
            qml.SQISW(wires=qubits[:2])

        complexity = 2
    elif gate_string == '010011' and n_qubits > 1:
        def gate():
            qml.SWAP(wires=qubits[:2])

        complexity = 1
    elif gate_string == '010100':
        def gate():
            qml.SX(wires=qubits[0])

        complexity = 1
    elif gate_string == '010101' and n_qubits > 1:
        def gate(theta):
            qml.SingleExcitation(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '010110' and n_qubits > 1:
        def gate(theta):
            qml.SingleExcitationPlus(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '010111' and n_qubits > 1:
        def gate(theta):
            qml.SingleExcitationMinus(theta, wires=qubits[:2])

        n_params = 1
        complexity = 3
    elif gate_string == '011000' and n_qubits > 2:
        def gate():
            qml.Toffoli(wires=qubits[:3])

        complexity = 2
    elif gate_string == '011001':
        def gate(theta):
            qml.U1(theta, wires=qubits[0])

        n_params = 1
        complexity = 1
    elif gate_string == '011010':
        def gate(theta, phi):
            qml.U2(theta, phi, wires=qubits[0])

        n_params = 2
        complexity = 1
    elif gate_string == '011011':
        def gate(theta, phi, delta):
            qml.U3(theta, phi, delta, wires=qubits[0])

        n_params = 3
        complexity = 2
    elif gate_string == '011101' and n_qubits > 3:
        def gate(theta):
            qml.DoubleExcitation(theta, wires=qubits[:4])

        n_params = 1
        complexity = 4
    elif gate_string == '011110' and n_qubits > 3:
        def gate(theta):
            qml.DoubleExcitationPlus(theta, wires=qubits[:4])

        n_params = 1
        complexity = 4
    elif gate_string == '011111' and n_qubits > 3:
        def gate(theta):
            qml.DoubleExcitationMinus(theta, wires=qubits[:4])

        n_params = 1
        complexity = 4
    else:
        gate = None
    return gate, n_params, complexity


def genome_to_circuit(genome, n_qubits, n_gates, initial_state):
    """
    # Set string and transform it into a quantum circuit.

    """
    gates = []
    n_params = []
    total_complexity = 0
    gene_length = len(genome) // n_gates
    for i in range(n_gates):
        gene = genome[i * gene_length: (i + 1) * gene_length]
        gate, n, complexity = string_to_gate(gene, n_qubits)
        if gate is not None:
            gates.append(gate)
            n_params.append(n)
            total_complexity += complexity

    def circuit(params):
        qml.BasisState(initial_state, wires=range(n_qubits))
        param_counter = 0
        for gate, n in zip(gates, n_params):
            p = params[param_counter: param_counter + n]
            gate(*p)
            param_counter += n

    total_params = sum(n_params)
    return circuit, total_params, total_complexity


class Agent:
    """
    1. Create a population of agents
    2. Evaluate the fitness of each one
    3. Select between the best agents
    4. Breed between them
    5. Random mutate genes of the genome of the agent
    """

    def __init__(self, length):
        self.string = ''.join(str(random.randint(0, 1)) for _ in range(length))
        self.fitness = -1
        self.energy = 0

    def __str__(self):
        return ' String: ' + str(self.string) + ' Fitness: ' + str(np.round(self.fitness, 6)) + ' Energy: ' + str(
            self.energy)


def init_agents(population, length):
    return [Agent(length) for _ in range(population)]


def fitness(agents, n_qubits, n_gates, circuit_to_cost_fn, initial_state):
    for agent in agents:
        genome = agent.string
        circuit, n_params, total_complexity = genome_to_circuit(genome, n_qubits, n_gates, initial_state)
        cost_fn = circuit_to_cost_fn(circuit)
        initial_theta = np.zeros(n_params)
        energy, _ = run_vqe(cost_fn, initial_theta)
        agent.fitness = -energy - total_complexity * 0.0001
        agent.energy = energy
    return agents


def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
    print('\n'.join(map(str, agents)))
    # Natural selection
    kill_param = 0.2  # take the top 20% of the individuals
    agents = agents[:int(kill_param * len(agents))]
    return agents


def crossover(agents, str_len, population):
    offspring = []
    for _ in range(int((population - len(agents)) / 2)):
        # TODO: don't breed parents that are the same
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)
        child1 = Agent(str_len)
        child2 = Agent(str_len)
        split = random.randint(0, str_len)
        child1.string = parent1.string[0:split] + parent2.string[split:str_len]
        child2.string = parent2.string[0:split] + parent1.string[split:str_len]
        offspring.append(child1)
        offspring.append(child2)
    agents.extend(offspring)
    return agents


def mutation(agents, str_len):
    chance_of_mutation = 0.20
    for agent in agents:
        for idx, param in enumerate(agent.string):
            if random.uniform(0.0, 1.0) <= chance_of_mutation:
                agent.string = agent.string[0:idx] + str(random.randint(0, 1)) + agent.string[idx + 1:str_len]
    return agents


def ga(population, generations, threshold, str_len, n_qubits, n_gates, circuit_to_cost_fn, initial_state=None):
    if initial_state is None:
        initial_state = np.zeros(n_qubits)
    agents = init_agents(population, str_len)
    for generation in range(generations):
        print("Generation: ", str(generation))
        agents = fitness(agents, n_qubits, n_gates, circuit_to_cost_fn, initial_state)
        agents = selection(agents)
        agents = crossover(agents, str_len, population)
        agents = mutation(agents, str_len)
        if any(agent.fitness >= threshold for agent in agents):
            print("\U0001F986 Threshold has been met! Winning genome: ", agents[0].string)
            return agents[0].string
    return agents[0].string
