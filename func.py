import numpy as np
import pandas as pd
import cvxopt
from cvxopt import glpk
import os
from scipy import stats
from scipy.stats import ortho_group
from heapq import heapify, heappush, heappop
import networkx as nx

class Map():
    def __init__(self):
        pass

    def make_map_with_M(self, mu, sigma, M, OD_ori=None, mu2=None, sigma2=None, phi_bi=None):
        self.mu = mu
        self.sigma = sigma
        self.M = M
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.phi_bi = phi_bi
        self.G = convert_map2graph(self)
        if OD_ori is not None:
            self.b, self.r_0, self.r_s = generate_b(self.M.shape[0], OD_ori[0], OD_ori[1])
            self.dij_cost, self.dij_path, self.dij_onehot_path = dijkstra(self.G, self.r_0, self.r_s)

    def make_map_with_G(self, mu, sigma, G, OD_true, mu2=None, sigma2=None, phi_bi=None):
        self.mu = mu
        self.sigma = sigma
        self.r_0, self.r_s = OD_true[0], OD_true[1]
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.phi_bi = phi_bi
        self.G = G
        self.M = None
        self.b = None

    def update_OD(self, OD_ori):
        self.b, self.r_0, self.r_s = generate_b(self.M.shape[0], OD_ori[0], OD_ori[1])
        self.dij_cost, self.dij_path, self.dij_onehot_path = dijkstra(self.G, self.r_0, self.r_s)

    def update_mu(self, new_mu):
        self.mu = new_mu
        self.G = update_graph_weight(self.G, new_mu)

    def extract_map(self, map_index=0, r_0=1, r_s=15):
        self.M, self.idx, self.mu = extract_map(map_index)
        self.sigma = generate_sigma3(self.mu, 0.7)
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.b, self.r_0, self.r_s = generate_b(self.M.shape[0], r_0, r_s)
        self.G = None

    def generate_loop_map(self, n_loop=2):
        self.M, self.idx, self.b, self.mu, self.r_0, self.r_s = generate_loop_map(loops=n_loop, let=200)
        self.sigma = 0.1 * self.mu * np.diag(np.random.rand(np.size(self.mu), 1).transpose()[0])
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.norm_x = cvxopt_glpk_minmax(self.mu, self.M, self.b)
        self.norm_path = np.flatnonzero(self.norm_x)

    def generate_map(self, n=1):
        self.M, self.idx = generate_A(n)
        self.b, self.r_0, self.r_s = generate_b(self.M.shape[0], 1, 3)
        self.mu = np.array([10, 10.1, 10.2, 20]).T
        self.sigma = np.array([[2,-1,1,0],[-1,2,0,0],[1,0,1,0],[0,0,0,1]])
        self.n_node = self.M.shape[0]
        self.n_link = self.M.shape[1]
        self.G = convert_map2graph(self)
        self.dij_cost, self.dij_path, self.dij_onehot_path = dijkstra(self.G, self.r_0, self.r_s)

class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'
    The 'smallest' method can be used to return the object with lowest
    priority, and 'get' also removes it.
    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def get(self):
        """Return the item with the lowest priority and remove it.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()

    def empty(self):
        return True if not self._heap else False
    
def generate_A(n):                              # n: # of "loop" structure
    A = np.zeros((n+2,2*n+2))                   # n+2: # of nodes; 2n+2: # of links
    A[0,0] = 1
    A[1,0] = -1
    A[0,2*n+1] = 1
    A[n+1,2*n+1] = -1
    for i in range(0,n):
        A[i+1,2*i+1] = 1
        A[i+1,2*i+2] = 1
        A[i+2,2*i+1] = -1
        A[i+2,2*i+2] = -1

    A_idx = np.arange(1,2*n+3)                  # true index of links
    return A, A_idx

def generate_b(n_node, origin, destination):    # o and d count from 1, while store from 0
    b = np.zeros(n_node)

    r_0 = origin-1
    r_s = destination-1

    b[r_0] = 1
    b[r_s] = -1

    return b.reshape(-1,1), r_0, r_s

def generate_mu(n_link, mu_scaler=10):
    mu = mu_scaler*np.ones(n_link)
    # mu[0][np.random.randint(1,n_link-1)] += 0.1
    mu[-1] = (n_link/2)*mu_scaler

    # mu = np.random.rand(1,n_link)
    # mu[-1] = n_link/4.5
    return mu.reshape(-1,1)

def generate_sigma(n_link, sigma_scaler=1):
    D = sigma_scaler*np.diag(np.random.rand(n_link))
    U = ortho_group.rvs(dim=n_link)
    sigma = np.matmul(np.matmul(U.T,D),U)
    return sigma

def generate_map(n, origin=1, destination=None):
    if destination == None:
        destination = n+2
    n_node = n+2
    n_link = 2*n+2
    A, A_idx = generate_A(n)
    b, r_0, r_s = generate_b(n_node,origin,destination)
    return A, A_idx, b, r_0, r_s, n_link


# A, A_idx, b, r_0, r_s = generate_map(2)
# mu = generate_mu(6)
# sigma = generate_sigma(6)
# print(A)
# print(A_idx)
# print(b)
# print(mu)
# print(sigma)
# print(r_0)
# print(r_s)

def cvxopt_glpk_minmax(c, A, b, x_min=0, x_max=1):
    dim = np.size(c,0)

    x_min = x_min * np.ones(dim)
    x_max = x_max * np.ones(dim)
    G = np.vstack([+np.eye(dim),-np.eye(dim)])
    h = np.hstack([x_max, -x_min])
    # G = -np.eye(dim)
    # h = x_min.T

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')
    G = cvxopt.matrix(G,tc='d')
    h = cvxopt.matrix(h,tc='d')
    # sol = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    _,x = glpk.ilp(c,G,h,A,b,options={'msg_lev':'GLP_MSG_OFF'})

    return np.array(x)

def cvxopt_glpk_binary(c, G, h, A, b):
    dim = np.size(c,0)

    B = {i for i in range(dim)}

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')
    G = cvxopt.matrix(G,tc='d')
    h = cvxopt.matrix(h,tc='d')
    # sol = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    _,x = glpk.ilp(c,G,h,A,b,B=B,options={'msg_lev':'GLP_MSG_OFF'})

    return np.array(x)

def update_map(A, b, link, curr_node, next_node):
    A_temp = np.delete(A,link,axis=1)
    b_temp = np.copy(b)
    b_temp[curr_node] = 0
    b_temp[next_node] = 1
    return A_temp, b_temp

def update_param(mu, sigma, link):
    mu_1 = np.delete(mu,link,axis=0)
    mu_2 = mu[link][0]
    mu_sub = {1:mu_1, 2:mu_2}

    sigma_11 = np.delete(np.delete(sigma,link,axis=1),link,axis=0)
    sigma_12 = np.delete(sigma[:,link],link,axis=0).reshape(-1,1)
    sigma_21 = np.delete(sigma[link,:],link).reshape(1,-1)
    sigma_22 = sigma[link,link]
    sigma_sub = {11:sigma_11, 12:sigma_12, 21:sigma_21, 22:sigma_22}

    sigma_con = sigma_11-np.matmul(sigma_12,sigma_21)/sigma_22

    return mu_sub, sigma_sub, sigma_con

def update_mu(mu_sub, sigma_sub, sample):
    return mu_sub[1]+(sample-mu_sub[2])/sigma_sub[22]*sigma_sub[12]

def calc_exp_gauss(mu, sigma):
    sigma_diag = np.diag(sigma).reshape(-1,1) if type(sigma) is np.ndarray else sigma
    exp_mu = np.exp(mu+sigma_diag/2)
    return exp_mu

def calc_bi_gauss(phi, mu1, mu2):
    return phi*mu1+(1-phi)*mu2

def extract_map(map_id):
    table_paths = ['Maps/SiouxFalls/SiouxFalls_network.xlsx',
                'Maps/SiouxFalls/SiouxFalls_network_copy.xlsx',
                'Maps/Anaheim/Anaheim_network.xlsx',
                "Maps/Winnipeg/winnipeg_network.xlsx",
                "Maps/Barcelona/Barcelona_network.xlsx"]

    raw_map_data = pd.read_excel(table_paths[map_id])

    origins = raw_map_data['From']
    destinations = raw_map_data['To']
    n_node = max(origins.max(), destinations.max())
    n_link = raw_map_data.shape[0]

    A = np.zeros((n_node,n_link))
    for i in range(n_link):
        A[origins[i]-1,i] = 1
        A[destinations[i]-1,i] = -1

    A_idx = np.arange(1,n_link+1)

    mu = np.array(raw_map_data['Cost']).reshape(-1,1)

    return A, A_idx, mu

def add_noise_to_mu(mu, nu=0.05):
    n_link = np.size(mu)
    sigma = nu*mu#*np.random.rand(n_link,1)

    mu_noise = np.zeros((n_link,1))
    for i in range(n_link):
        while mu_noise[i] <= 0:
            mu_noise[i] = np.random.normal(mu[i],sigma[i])

    return mu_noise

def generate_grid_map(dim):
    n_node = (dim+1)**2
    n_link = 2*dim*(dim+1)

    b, r_0, r_s = generate_b(n_node, 1, n_node)

    A = np.zeros((n_node,n_link))
    for i in range(dim+1):
        for j in range(dim):
            A[i*(dim+1)+j][i*dim+j] = 1
            A[i*(dim+1)+1+j][i*dim+j] = -1

    n_half_link = dim*(dim+1)
    for i in range(n_half_link,n_link):
        A[i-n_half_link][i] = 1
        A[i-dim**2+1][i] = -1

    A_idx = np.arange(n_link)

    mus = 10*np.ones(2*dim)
    # mus = np.random.uniform(5,15,2*dim)
    mu = np.zeros((n_link,1))
    for i in range(dim):
        for j in range(i,dim*(dim+1),dim):
            mu[j] = mus[i]
        for j in range((dim+1)*(dim+i),(dim+1)*(dim+1+i)):
            mu[j] = mus[i+dim]

    # mu = add_noise_to_mu(mu,0.03)

    return A, A_idx, b, mu, r_0, r_s

def generate_loop_map(loops, let=200):
    n_node = loops+1
    n_link = 2*loops

    b, r_0, r_s = generate_b(n_node, 1, n_node)

    A = np.zeros((n_node,n_link))
    for i in range(loops):
        A[i][2*i] = 1
        A[i+1][2*i] = -1
        A[i][2*i+1] = 1
        A[i+1][2*i+1] = -1

    A_idx = np.arange(n_link)

    mu = (let/loops)*np.ones(n_link).reshape(-1,1)

    return A, A_idx, b, mu, r_0, r_s

def generate_line_map(lines):
    n_node = 4
    n_link = lines+4

    b, r_0, r_s = generate_b(n_node, 1, n_node)

    A = np.zeros((n_node,n_link))
    A[0,0:2] = 1
    A[1,0:2] = -1
    A[2,2:4] = 1
    A[3,2:4] = -1
    A[1,4:] = 1
    A[2,4:] = -1


    A_idx = np.arange(n_link)

    mu = 10*np.ones(n_link).reshape(-1,1)

    return A, A_idx, b, mu, r_0, r_s

def generate_cov(mu, nu):
    n_link = np.size(mu)

    sigma = nu*mu*np.random.rand(n_link,1)

    n_sample = n_link
    samples = np.zeros((n_link,n_sample))

    for i in range(np.shape(samples)[0]):
        for j in range (np.shape(samples)[1]):
            # while samples[i][j] <= 0:
            samples[i][j] = np.random.normal(mu[i],sigma[i])

    cov = np.cov(samples)

    return cov

def generate_cov1(mu, nu, factors):         #factors up, corr down
    n_link = np.size(mu)

    W = np.random.randn(n_link,factors)
    S = np.dot(W,W.T) + np.diag(np.random.rand(1,n_link))
    corr = np.matmul(np.matmul(np.diag(1/np.sqrt(np.diag(S))),S),np.diag(1/np.sqrt(np.diag(S))))

    sigma = nu*mu#*np.random.rand(n_link,1).reshape(-1,1)
    # sigma = nu * np.random.random(n_link).reshape(-1,1)

    sigma = np.matmul(sigma,sigma.T)

    cov = sigma*corr

    return corr, sigma, cov

def judge(corr):
    for i in range(np.shape(corr)[0]):
        if np.sum(np.abs(corr[i]))-np.abs(corr[i,i]) > 1:
            return 1
    return 0

def generate_cov2(mu, nu, mean, std):
    n_link = np.size(mu)
    sigma = nu*mu
    sigma = np.matmul(sigma,sigma.T)

    corr = np.ones((n_link,n_link))
    while judge(corr):
        for i in range(n_link):
            corr_row = np.ones((1,n_link))
            while (np.sum(np.abs(corr_row))-np.abs(corr_row[0,i])) > 1:
                corr_row = np.random.normal(mean, std, (1,n_link))
            corr_row[0,i] = 1
            corr[i,:] = corr_row
        corr = (corr + corr.T)/2
    cov = sigma*corr

    return cov

def generate_cov3(mu, nu):
    n_link = np.size(mu)

    D = np.diag(np.random.rand(n_link))
    U = ortho_group.rvs(dim=n_link)
    S = np.matmul(np.matmul(U.T, D), U)
    corr = np.matmul(np.matmul(np.diag(1 / np.sqrt(np.diag(S))), S), np.diag(1 / np.sqrt(np.diag(S))))

    sigma = nu * mu  # *np.random.rand(n_link,1).reshape(-1,1)
    sigma = np.matmul(sigma, sigma.T)
    cov = sigma * corr

    return corr, sigma, cov


def generate_sigma2(mu, nu):
    sigma = nu * mu.max()/mu

    sigma2 = np.diag(np.square(sigma).reshape(-1))

    return sigma2

def generate_sigma3(mu, nu):
    sigma = np.random.uniform(0, nu) * mu

    sigma2 = np.diag(np.square(sigma).reshape(-1))

    return sigma2


def get_let_path(mu,A,b):
    sol = cvxopt_glpk_minmax(mu,A,b)
    if sol.all() == None:
        return None, None

    else:
        selected_links = list(np.where(sol == 1)[0])

        num_sel_links = len(selected_links)
        sorted_links = []
        node = np.where(b==1)[0].item()
        while num_sel_links != len(sorted_links):
            for link in selected_links:
                if A[node,link] == 1:
                    sorted_links.append(link)
                    node = np.where(A[:,link]==-1)[0].item()
                    selected_links.remove(link)
                    break
        sorted_links = [link+1 for link in sorted_links]

        cost = np.dot(sol.T,mu).item()

        return sorted_links, cost

def get_let_first_step(mu,A,b):
    sol = cvxopt_glpk_minmax(mu,A,b)
    selected_links = list(np.where(sol == 1)[0])

    node = np.where(b==1)[0].item()
    for link in selected_links:
        if A[node,link] == 1:
            first_step = link
            break

    cost = np.dot(sol.T,mu).item()

    return first_step, cost

def generate_cov_log(mu_ori, nu):
    n_link = np.size(mu_ori)

    sigma = nu*mu_ori*np.random.rand(n_link,1)

    sigma_log = np.log(np.divide(sigma**2,mu_ori**2)+1)
    mu_log = np.log(mu_ori)-0.5*sigma_log

    n_sample = n_link
    samples = np.zeros((n_link,n_sample))

    for i in range(np.shape(samples)[0]):
        samples[i] = np.random.lognormal(mu_log[i],np.sqrt(sigma_log[i]),(1,np.shape(samples)[1]))

    cov_ori = np.cov(samples)

    return cov_ori

def calc_logGP4_param(mu_ori, cov_ori):
    cov_log = np.log(cov_ori/np.dot(mu_ori,mu_ori.T)+1)
    mu_log = np.log(mu_ori)-0.5*np.diag(cov_log).reshape(-1,1)

    return mu_log, cov_log

def generate_biGP_mus(phi_bi, mu1, mu2, sigma1, sigma2, n_W, method='cholesky'):
    rng = np.random.default_rng()
    if type(mu1) is np.ndarray:
        mus1 = rng.multivariate_normal(mu1.reshape(-1), sigma1, n_W, method=method)
        mus2 = rng.multivariate_normal(mu2.reshape(-1), sigma2, n_W, method=method)
        # mus1 = np.random.multivariate_normal(mu1.reshape(-1), sigma1, n_W)
        # mus2 = np.random.multivariate_normal(mu2.reshape(-1), sigma2, n_W)
        dim = mu1.size
    else:
        mus1 = np.random.normal(mu1, np.sqrt(sigma1), [n_W,1])
        mus2 = np.random.normal(mu2, np.sqrt(sigma2), [n_W,1])
        dim = 1

    phi1 = np.where(np.random.rand(n_W, dim) < phi_bi, 1, 0)
    phi2 = np.ones(phi1.shape)-phi1
    mus = np.multiply(phi1,mus1) + np.multiply(phi2,mus2)

    return mus

def generate_mus(mymap, n_W, model='G', method='cholesky'):
# return shape = N*n_W
    rng = np.random.default_rng()
    if model == "G":
        mus = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.sigma, n_W, method=method)
        # mus = np.random.multivariate_normal(mymap.mu.reshape(-1), mymap.sigma, n_W)
    elif model == "log":
        mus = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.sigma, n_W, method=method)
        # mus = np.random.multivariate_normal(mymap.mu.reshape(-1), mymap.sigma, n_W)
        mus = np.exp(mus)
    elif model == "bi":
        mus = generate_biGP_mus(mymap.phi_bi, mymap.mu, mymap.mu2, mymap.sigma, mymap.sigma2, n_W)
    return mus.T

def generate_mu_(mymap, n):
    mu = np.random.normal(mymap.mu[n], mymap.sigma[n][n])
    return mu

def generate_dependent_mus(mymap, n):
    sigma = np.zeros((mymap.mu.size, 1))
    for i in range(mymap.mu.size):
        sigma[i] = mymap.sigma[i][i]
    mus = np.random.normal(mymap.mu.reshape(-1), sigma.reshape(-1), (n, mymap.mu.size))
    return mus

def convert_node2onehot(path, G):
    link_ids = []
    node_pairs=zip(path[0:],path[1:])

    for u,v in node_pairs:
        edge = sorted(G[u][v], key=lambda x:G[u][v][x]['weight'])
        link_ids.append(G[u][v][edge[0]]['index'])

    onehot = np.zeros(G.size())
    onehot[link_ids] = 1
    onehot = onehot.reshape(-1,1)

    return link_ids, onehot

def convert_map2graph(mymap):
    G = nx.DiGraph()

    for i in range(mymap.M.shape[1]):
        start = np.where(mymap.M[:,i]==1)[0].item()
        end = np.where(mymap.M[:,i]==-1)[0].item()
        G.add_edge(start, end, weight=mymap.mu[i].item(), index=i)

    return G

def update_graph_weight(G, new_mu):
    G_new = G.copy()
    for u,v,k,d in G.edges(data=True, keys=True):
        G_new[u][v][k]['weight'] = new_mu[d['index']].item()
    return G_new

def remove_graph_edge(G, e_id):
    G_new = G.copy()
    for u,v,k,d in G.edges(data=True, keys=True):
        if d['index'] == e_id:
            G_new.remove_edge(u,v,k)
        elif d['index'] > e_id:
            G_new[u][v][k]['index'] -= 1
    return G_new

def find_next_node(mymap, curr_node, link_idx):
    for _, next_node, d in mymap.G.out_edges(curr_node, data=True):
        if d['index'] == link_idx:
            return next_node


def dijkstra(G, start, end, mus=None):
    if not G.has_node(start) or not G.has_node(end):
        return -1, None, None

    cost = {}
    for node in G.nodes():
        cost[node] = float('inf')
    cost[start] = 0
    prev_node = {start: None}
    prev_edge = {start: None}
    PQ = priority_dict(cost)

    while bool(PQ):
        curr_node = PQ.get()

        if curr_node == end:
            break

        for _, next_node, d in G.out_edges(curr_node, data=True):
            if next_node in PQ:
                alt = cost[curr_node] + (d['weight'] if mus is None else mus[d['index']].item())
                if alt < cost[next_node]:
                    cost[next_node] = alt
                    prev_node[next_node] = curr_node
                    prev_edge[next_node] = d['index']
                    PQ[next_node] = alt

    if curr_node == end and end in prev_node:
        path_cost = cost[end]
        path = []
        while curr_node != start:
            path.append(prev_edge[curr_node])
            curr_node = prev_node[curr_node]
        path.reverse()

        onehot = np.zeros(G.size())
        onehot[path] = 1
        onehot = onehot.reshape(-1, 1)
        return path_cost, path, onehot
    else:
        return -1, None, None

def t_test(x, y, alternative='greater', alpha=0.05):
    t_stat, double_p = stats.ttest_ind(x,y,equal_var = False)

    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if t_stat > 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if t_stat < 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.

    return pval, pval<alpha

def modify_cov(cov, paths):
    temp = np.eye(cov.shape[0])

    for i in paths:
        for j in paths:
            temp[i-1,j-1] = 1
    
    return cov*temp

def generate_OD_pairs(mymap, n_pairs):
    def generate_OD(n_node):
        r_0 = np.random.randint(n_node) + 1
        r_s = np.random.randint(n_node) + 1
        while r_s == r_0:
            r_s = np.random.randint(n_node) + 1
        OD = [r_0, r_s]
        return OD

    OD_pairs = []
    count = 0

    while count < n_pairs:
        OD = generate_OD(mymap.n_node)
        while OD in OD_pairs or dijkstra(mymap.G, OD[0]-1, OD[1]-1)[0] == -1:
            OD = generate_OD(mymap.n_node)
        OD_pairs.append(OD)
        count += 1

    return OD_pairs

def generate_samples(mymap, S, model='G', decom_method='cholesky'):
    '''
    return: N*S matrix
    '''

    rng = np.random.default_rng()
    if model == "G":
        samples = rng.multivariate_normal(mymap.mu.reshape(-1), mymap.sigma, S, method=decom_method)
    elif model == "log":
        sigma2 = mymap.sigma
        mu = mymap.mu
        mu_log, cov_log = calc_logGP4_param(mu, sigma2)
        samples = rng.multivariate_normal(mu_log.reshape(-1), cov_log, S, method=decom_method)
        samples = np.exp(samples)
    # elif model == "bi":
    #     samples = generate_biGP_samples(mymap.phi_bi, mymap.mu, mymap.mu2, mymap.cov, mymap.cov2, S, method=decom_method)
    return samples.T

# link_path从0开始
def get_cost_from_path(mymap, link_path):
    link_path = np.array(link_path)
    cost = 0
    for link in link_path:
        cov2_log = np.log(np.diag(mymap.sigma)[link] / mymap.mu[link] ** 2 + 1)
        mu_log = np.log(mymap.mu[link] ** 2 / (np.sqrt(np.diag(mymap.sigma)[link] + mymap.mu[link] ** 2)))
        cost += np.random.lognormal(mu_log, np.sqrt(cov2_log))
    return cost

