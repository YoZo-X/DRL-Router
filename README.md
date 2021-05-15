# DRL-Router

This is an implementation of DRL-Router on Python 3, Numpy, and Networkx. DRL-Router is a method based on distributional reiforcement learning for RSP problem.

## Table of Contents
- [DRL-Router](#drl-router)
  - [Requirements](#requirements)
  - [How to use](#how_to_use)
  - [Template](#template)

### 1.Requirements
  Python 3.7, numpy, cvxopt, scipy, heapq, networkx and other common packages.

### 2. How to use
  **step 1**: Create a Map, then extract transcanction data or make a data by yourself;

  **step 2**: Create a Xtate based on the Map that is created on **step 1**;

  **step 3**: Create a Agent of DRL-Router based on the Xtate that is created on **step 2**;

  **step 4**: Configure the parameter of the Agent that is created on **step 3**, where K(number of samples), lr_rate(learning rate), v_min(range of distribution) and termination parameters are necessary;

  **step 5**: We need use dijkstra to pretrain the Agent, which is a warm start for DRL-Router. We suggest turning on the dynamic learning rate(dynamic_lr = 1) during pre-training and running 1000 episodes;

   **step 6**: We can finally start the training of the Agent, we need to set the training parameters num_iterations, obj(define RSP problem) and parameter(different parameter for different RSP problem). When the training was over we got a *Policy*. The more training times, the more accurate the *Policy* results will be.

### 3. Template
  The following is an example of how to configure a DRL-Router：
  ```Python
  import DRL_C51
  import func

  Map_Name = "SiouxFalls"

  Map_id = {"SiouxFalls": 0,
            "Anaheim": 2,
            "Winnipeg": 3,
            "Barcelona": 4}

  Map_1 = func.Map()
  Map_1.extract_map(Map_id[Map_Name])
  Map_1.G = func.convert_map2graph(Map_1)

  X = DRL_C51.Xtates(Map_1, num_atoms=51)
  agent = DRL_C51.DRL_Agent(X, Map, 15)
  agent.update_V(-200, 0)
  agent.dynamic_lr = 1
  agent.lr_rate = 0.01
  agent.K = 5
  agent.train_dijkstra(1000)
  agent.dynamic_lr = 0
  agent.lr_rate = 0.1
  agent.train_IS(2000, parameter=40, obj="mean-std")
  print("-----------shortest path-------------")
  agent.find_shortest_path(1, True)
  print("-----------C51 path-------------")
  agent.find_path(1, 40, "mean-std", True)
  ```
