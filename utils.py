import networkx as nx
import numpy as np
import operator
import time
import matplotlib.pyplot as plt
np.random.seed(4)


def generate_power_law_dg_sequence(lambda_val, num, K0 = 2/3, m=1):
    K = int(num**(K0))
    while True:  
        s=[]
        while len(s)<num:
            nextval = int(nx.utils.powerlaw_sequence(1,lambda_val)[0]) #100 nodes, power-law exponent 2.5
            if nextval!=0:
                s.append(nextval)
        s.append(m)
        s.append(K)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G=nx.Graph(G) # remove parallel edges
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(components[0])
    
    print()
    print("STATISTIQUES")
    print()
    print("Size of largest connected component = {0:d}".format(len(lcc)), "propotion: ", 100*len(lcc)/num)
    print()
    print_graph_stats("Scale free model", G)
    # Let's check the graph stats
    print()
    print_graph_stats("Largest connected component", lcc)
    print()
    plt.figure(figsize = (12,6))
    counts = np.bincount(s)
    mask = (counts > 0)
    plt.plot(np.arange(len(counts))[mask], counts[mask] / counts.sum(), "o", label=r"degree sequence for the graph")
    x = np.arange(1, len(counts))
    plt.plot(x, np.power(x, -lambda_val), label=r"$y = -\lambda x$")
    plt.xlabel(r"Degree $k$")
    plt.xscale("log")
    plt.ylabel(r"Probability $P(k)$")
    plt.yscale("log")
    plt.title(r"Logaritmique plot with $  N = {0:d}, \quad \lambda = {1:.2f}$".format(int(num), lambda_val))
    plt.legend(loc="best")
    plt.show()
    plt.savefig('Log P with degree K.pdf')
    return G, lcc

def print_graph_stats(title, g):
    print("Simple stats for: " + title)
    print("# of nodes: " + str(len(g.nodes())))
    print("# of edges: " + str(len(g.edges())))
    print("Is graph connected? " + str(nx.is_connected(g)))



class SIR:
    def __init__(self, g, beta, mu, Tmax = 30, indect_start = 0.001):
        self.g = g
        self.beta = beta #transmission rate
        self.mu = mu # recovery rate
        self.indect_start = indect_start
        self.Tmax = Tmax
        
    def run(self, seed=[], num_steps = 1, sentinels = [], immunization_rate = 0.0, immunized_nodes = []):
        # Immunize nodes according to the set immunization rate.
        if len(immunized_nodes) == 0:
            immunized = set(np.random.choice(self.g.nodes(), 
                                             size=int(immunization_rate*len(self.g.nodes())), 
                                             replace=False))
        else:
            immunized = immunized_nodes
        
        
        # If there is no seed, just choose a random node in the graph.
        if len(seed) == 0:
            nombre_personne_depart = int(self.indect_start*len(list(set(self.g.nodes()).difference(immunized))))
            seed = np.random.choice(list(set(self.g.nodes()).difference(immunized)), nombre_personne_depart, replace=False)
            
            #print(seed)
        
        I_set = set(seed)
        #print(I_set)
        S_set = set(self.g.nodes()).difference(I_set).difference(immunized)
        R_set = set()
        
        number_of_personn_infected_sofar = {i:0.0 for i in self.g.nodes()}
        number_of_timepeoplearestayinfected_sofar = {i:0.0 for i in self.g.nodes()}
        
        
        t = 0
        
        StoI = set(seed)
        ItoR = set()
        
        sentinels_t = {}
        for sen in sentinels:
            sentinels_t[sen] = 0
        
        while (len(I_set) > 0 and t < self.Tmax):
            I_set_old = I_set.copy()
            # Let's infect people! 
            for i in I_set.copy():
                #print(len(set(self.g.neighbors(i)).intersection(S_set).copy()))
                ntot = len(set(self.g.neighbors(i)).intersection(S_set).copy())
                for s in set(self.g.neighbors(i)).intersection(S_set).copy():
                    if np.random.uniform() < self.beta:
                        S_set.remove(s)
                        I_set.add(s)
                        StoI.add(s)
                        number_of_personn_infected_sofar[i]+=1
                        # Record t for sentinels
                        if sentinels_t.get(s) != None:
                            sentinels_t[s] = t
                            
                number_of_timepeoplearestayinfected_sofar[i] += 1
                #print(t, i, number_of_timepeoplearestayinfected_sofar[i], number_of_personn_infected_sofar[i], ntot)
            
                #print(t, number_of_personn_infected_sofar[i] )
                
                        
                # Will infected person recover?
                if np.random.uniform() < self.mu:
                    I_set.remove(i)
                    R_set.add(i)
                    ItoR.add(i)
            
                
                
    
            t += 1
            nbre_jour_rester_infecter = int(1/self.mu)
            all_K = []
            for k in I_set_old:
                val2 = max(1, nbre_jour_rester_infecter - number_of_timepeoplearestayinfected_sofar[k]+1)
                val = min(len(list(self.g.neighbors(k))), number_of_personn_infected_sofar[k] *val2)
                #print(t, val, number_of_personn_infected_sofar[k], len(list(self.g.neighbors(k))))
                all_K.append(val)
            #print(t, np.mean(all_K))
            #print(t, np.mean([number_of_personn_infected_sofar[k]* for k in I_set_old]), np.mean([number_of_timepeoplearestayinfected_sofar[k] for k in I_set_old]) )
            if t % num_steps == 0 or len(I_set) == 0:
                yield({'t': t, 'S':S_set, 'I':I_set, 'R':R_set, 'StoI':StoI, 'ItoR':ItoR, 'sentinels': sentinels_t, 'reproductive_numbe':  np.mean(all_K)})


def get_temporal_plot(graph_l, b, m0, Tmax, indect_start, g1, g2):
    if graph_l == "The full graph":
        graph = g1
    else:
        graph = g2
    m = 1./m0
    
    print()
    print("Constante epedemie tau x r : ", 1/m*b)
    print()
    sir = SIR(graph, beta = b, mu = m, Tmax=Tmax, indect_start=indect_start)
    res = []
    res2 = []
    final_rs = []
    for r in sir.run(num_steps=1):
        n= len(r['S'])+ len(r['I']) +len(r['R'])
        res.append([len(r['I'])/n,1-len(r['R'])/n ])
        res2.append([r['reproductive_numbe'], 1 ])
    value_fin = len(r['R'])*100/len(graph.nodes())
    plt.figure(figsize=(36,24))
    ax = plt.subplot(121)
    ax.set_prop_cycle('color', ['green', 'blue'])
    # Plotting the epidemic curve.
    plt.plot(res)
    # Plot the results.
    plt.title("Epidemic curves for simulations with final "+str(value_fin)[:4] + "% of removed")
    plt.legend(['Infected', 'Removed'])
    plt.xlabel("t with r and tau equal to " + str(b) + " "+ str(m) + " "+ str(indect_start))
    plt.ylabel("Proportion of people infected", color="green")
    ax2 = ax.twinx()
    ax2.set_ylim(1, 0)
    ax2.set_ylabel('1-Proportion of people removed', color="blue")
    ax = plt.subplot(122)
    ax.set_prop_cycle('color', ['orange', 'red'])
    plt.plot(res2)
    plt.title("Basic reproductive numbe with time")
    plt.legend(['R0 value', 'thr'])
    plt.xlabel("t")
    plt.ylabel("R0")
    plt.savefig("Epidemic curve over time if nothing happens.png")
    
    print()
    print("Final removed pourcentage :", str(value_fin)[:4])

    
    return res, res2

def get_random_plot(graph, b, m, Tmax, indect_start, N, immunization_rates):

    start = time.time()
    i_sir = SIR(graph, beta = b, mu = m, Tmax=Tmax, indect_start=indect_start)
    final_rs = {}
    final_rs0 = {}

    for ir in immunization_rates:
        final_rs[ir] = []
        final_rs0[ir] = []
        for i in range(0,N):
            simulation_steps = [[len(r['S']), len(r['I']), len(r['R']), r['reproductive_numbe']] for r in i_sir.run(num_steps=1, 
                                                                                           immunization_rate = ir)]
            final_rs.get(ir).append(simulation_steps[len(simulation_steps)-1][2]*100/len(graph.nodes()))
            #print(simulation_steps[0][-1])
            moyenne_ro = []
            for k in range(len(simulation_steps)):
                moyenne_ro.append(simulation_steps[k][3])
            final_rs0.get(ir).append(np.mean(moyenne_ro))
    sorted_ir = sorted(final_rs.items(), key=operator.itemgetter(0))
    sorted_ir0 = sorted(final_rs0.items(), key=operator.itemgetter(0))


    print("Job done in: ", time.time() - start)


    irs = []
    oars = []
    oars2 = []

    for ir, values in sorted_ir:
        irs.append(ir)
        oars.append(np.mean(values))
    for ir, values in sorted_ir0:
        #irs.append(ir)
        oars2.append(np.mean(values))
    return irs, oars, oars2


def get_target_plot(graph, b, m, Tmax, indect_start, N, immunization_rates):
    start = time.time()

    ti_sir = SIR(graph, beta = b, mu = m, Tmax=Tmax, indect_start=indect_start)
    nodes_sorted_by_degree = sorted(nx.degree(graph), key=operator.itemgetter(1), reverse=True)
    final_rs = {}
    final_rs0 = {}
    for ir in immunization_rates:
        final_rs[ir] = []
        final_rs0[ir] = []
        # Immunize the M nodes with highest degree.
        immunized_nodes = []
        M = int(ir*len(nodes_sorted_by_degree))
        for i in range(M):
            immunized_nodes.append(nodes_sorted_by_degree[i][0])
        # Run the simulation 50 times and save the results.
        for i in range(0,N):
            simulation_steps = [[len(r['S']), len(r['I']), len(r['R']), r["reproductive_numbe"]] for r in ti_sir.run(num_steps=1, 
                                                                                            immunized_nodes = immunized_nodes)]
            final_rs.get(ir).append(simulation_steps[len(simulation_steps)-1][2]*100/len(graph.nodes()))
            moyenne_ro = []
            for k in range(len(simulation_steps)):
                moyenne_ro.append(simulation_steps[k][3])
            final_rs0.get(ir).append(np.mean(moyenne_ro))
    # Sort results and calculate the mean over the simulations to plot them.
    print("Job done in: ", time.time() - start)
    sorted_ir = sorted(final_rs.items(), key=operator.itemgetter(0))
    sorted_ir0 = sorted(final_rs0.items(), key=operator.itemgetter(0))
    t_irs = []
    t_oars = []
    t_oars0 = []
    for ir, values in sorted_ir:
        t_irs.append(ir)
        t_oars.append(np.mean(values))
    for ir, values in sorted_ir0:
        t_oars0.append(np.mean(values))
    
    return t_irs, t_oars, t_oars0, ti_sir


def get_aquitance_plot(graph, ti_sir, b, m, Tmax, indect_start, N, immunization_rates, K):
    sentinels = graph.nodes()
    sentinels_results = {}
    known_nodes = set(np.random.choice(graph.nodes(), size=int(K*len(graph.nodes())), replace=False))
    neighbors = set()
    for node in list(known_nodes):
        neighbors.update(set(graph.neighbors(node)))
    final_rs_k = {}
    final_rs0 = {}
    for ir in immunization_rates:
        final_rs_k[ir] = []
        final_rs0[ir] = []
        M = int(ir*len(neighbors))
        immunized_nodes_k = set(np.random.choice(list(neighbors), size=M, replace=False))
        for i in range(0,N):
            # Acquaintance immunization
            simulation_steps_k = [[len(r['S']), len(r['I']), len(r['R']),  r["reproductive_numbe"]] for r in ti_sir.run(num_steps=1, 
                                                                                            immunized_nodes = immunized_nodes_k)]
            final_rs_k.get(ir).append(simulation_steps_k[len(simulation_steps_k)-1][2]*100/len(graph.nodes()))
            moyenne_ro = []
            for k in range(len(simulation_steps_k)):
                moyenne_ro.append(simulation_steps_k[k][3])
            final_rs0.get(ir).append(np.mean(moyenne_ro))
    sorted_ir_k = sorted(final_rs_k.items(), key=operator.itemgetter(0))
    sorted_ir0 = sorted(final_rs0.items(), key=operator.itemgetter(0))
    irs2 = []
    oars_k = []
    oars_deg = []
    oars_sim = []
    t_oars01 = []
    for ir, values in sorted_ir_k:
        irs2.append((ir*len(neighbors))/len(graph.nodes()))
        oars_k.append(np.mean(values))
    for ir, values in sorted_ir0:
        t_oars01.append(np.mean(values))

    return irs2, oars_k, t_oars01


def final_plots_strat(graph_l, b, m0, Tmax, indect_start, g1, g2, N, immunization_rates, K):
    
    if graph_l == "The full graph":
        graph = g1
    else:
        graph = g2
    m = 1./m0
    
    print()
    print("Constante epedemie tau x r : ", 1/m*b)
    print()
    print("START RANDOM STRATEGY")
    print()
    irs, oars, oars2 = get_random_plot(graph, b, m, Tmax, indect_start, N, immunization_rates)
    print()
    print("END RANDOM STRATEGY")
    print()
    print("START Target STRATEGY")
    print()
    t_irs, t_oars, t_oars0, ti_sir = get_target_plot(graph, b, m, Tmax, indect_start, N, immunization_rates)
    print()
    print("END Target STRATEGY")
    print()
    print("START ACQUITANCE STRATEGY")
    print()
    irs2, oars_k, t_oars01 = get_aquitance_plot(graph, ti_sir , b, m, Tmax, indect_start, N, immunization_rates, K)
    print()
    print("END ACQUITANCE STRATEGY")
    print()
    
    plt.figure(figsize=(36, 24))
    plt.subplot(121).set_prop_cycle('color', ['yellow', 'blue', 'pink'])
    plt.title('Effectiveness of different immunization strategies')
    plt.xlabel('Immunization rate = proportion of the poulation vaccinated = proportion of vaccin needed')
    plt.ylabel('Proportion of the population removed')
    plt.plot(irs, oars)
    plt.plot(irs, t_oars)
    plt.plot(irs2, oars_k)
    plt.legend(['Random immunization', 'Targeted immunization', 'Acquitance immunization for K = 20%'], fontsize = 9)

    plt.subplot(122).set_prop_cycle('color', ['yellow', 'blue', 'pink', "red"])
    plt.title("Basic reproductive number for fifferent immunization strategies")
    plt.ylabel("R0")
    plt.xlabel('Immunization rate = proportion of the poulation vaccinated = proportion of vaccin needed')
    plt.plot(irs, oars2)
    plt.plot(irs, t_oars0)
    #plt.plot(irs, [1]*len(irs))
    plt.plot(irs2, t_oars01)
    plt.plot(irs, [1]*len(irs))
    plt.legend(['Random immunization', 'Targeted immunization', 'Acquitance immunization for K = 20%', "Threshold"], fontsize = 9)

    plt.savefig("plot final comparaison strategy.png")

    plt.show()