#import "@preview/theoframe:0.3.7": *
#show: theoframe-setup.with(theme: (style: "box", color: rgb("#067300")))

#show "s.t.": it => [_#(it)_]


The goal is to connect the Universal Reinforcement Learning (URL) framework with various formulations and applications. This involves exploring different mathematical representations, optimization techniques, and practical implementations to enhance the performance and adaptability of reinforcement learning algorithms across diverse environments.

We first define the categories of URL formulations for Q-learning methods. 


#definition(name: "Category of Markov decision processes")[
  The category of Markov decision processes, denoted as $cal(M)$, contains the following elements:
  + _object_: a Markov decision process $M = (S, A, Psi, P, R)$, where $S$ is the state space, $A$ is the action space, $Psi subset S times A$ is the set of the admissible state-action pairs, $P: S times A times A -> [0, 1]$ is the transition probability function, $R$ is the reward function.
  + _morphism_: these are the homomorphisms between Markov decision processes objects that preserve the transition probabilities and reward functions, s.t. for two MDPs $M_1$ and $M_2$, a morphism $xi: M_1 -> M_2$ is a map that satisfies the following conditions:
    1. $xi(S_1) subset S_2$ and $xi(A_1) subset A_2$.
    2. For all $(s, a) in Psi_1$, we have $(xi(s), xi(a)) in Psi_2$.
    3. For all $(s, a, s') in S_1 times A_1 times S_1$, we have $P_2(xi(s), xi(a), xi(s')) = P_1(s, a, s')$.
    4. For all $(s, a) in Psi_1$, we have $R_2(xi(s), xi(a)) = R_1(s, a)$.
]


#definition(name: "Category of action-value functions")[
  The category of action-value functions, denoted as $cal(Q)$, contains the following elements:
  + _object_: Q-function $f: S times A -> RR$, where $S$ is the state space, $A$ is the action space, and $RR$ is the set of real numbers.
  + _morphism_: a map $xi: Q_1 -> Q_2:f mapsto g compose f compose h^(-1)$ between two action-value functions $Q_1$ and $Q_2$ that preserves the structure of the functions, s.t. $(f_2 compose h)(s, a) = (g compose f_1)(s, a)$.
]


#definition(name:"Value evaluation functor")[
  The value evaluation functor, denoted as $V: M -> Q$, is a mapping from the category of Markov decision processes $M$ to the category of action-value functions $Q$. It assigns to each MDP $M = (S, A, Psi, P, R)$ an action-value function $Q_M: S times A -> RR$ defined by the Bellman equation:
  $
  Q_M(s, a) = R(s, a) + gamma sum_{s' in S} P(s' | s, a) max_(a' in A) Q_M(s', a')
  $
  for all $(s, a) in Psi$. The functor also maps morphisms between MDPs to morphisms between their corresponding action-value functions, preserving the structure of the functions.
]

The iterative process of finding the optimal Q in above definition can be interpreted as finding the fixed point in the corresponding coalgebra.


We can further define the category of world models, which are used to represent the environment in reinforcement learning. A world model captures the dynamics of the environment and can be used to simulate future states and rewards based on the current state and action.

#definition(name: "Category of world model")[
  The category of world models, denoted as $cal(W)$, contains the following elements:
  + _object_: a world model $W = (S, A, Psi, P, R)$, where $S$ is the state space, $A$ is the action space, $Psi subset S times A$ is the set of the admissible state-action pairs, $P: S times A times A -> [0, 1]$ is the transition probability function, and $R$ is the reward function.
  + _morphism_: these are the homomorphisms between world model objects that preserve the transition probabilities and reward functions, s.t. for two world models $W_1$ and $W_2$, a morphism $xi: W_1 -> W_2$ is a map that satisfies the following conditions:
    1. $xi(S_1) subset S_2$ and $xi(A_1) subset A_2$.
    2. For all $(s, a) in Psi_1$, we have $(xi(s), xi(a)) in Psi_2$.
    3. For all $(s, a, s') in S_1 times A_1 times S_1$, we have $P_2(xi(s), xi(a), xi(s')) = P_1(s, a, s')$.
    4. For all $(s, a) in Psi_1$, we have $R_2(xi(s), xi(a)) = R_1(s, a)$.
]

The category of world models share the same structure as the category of Markov decision processes, but they are specifically designed to represent the dynamics of the environment in reinforcement learning. The morphisms between world models allow for the transfer of knowledge and policies between different environments, enabling the development of more generalizable reinforcement learning algorithms.

We can define an adjunction between the category of world models and the category of Markov decision processes, which captures the relationship between the two categories and allows for the transfer of knowledge and policies between them.

#definition(name: "Adjunction between world models and MDPs")[
  An adjunction between the category of world models $cal(W)$ and the category of Markov decision processes $cal(M)$ consists of a pair of functors $F: cal(W) -> cal(M)$ and $G: cal(M) -> cal(W)$, along with a natural transformation $eta: id_{cal(W)} -> G compose F$ and a natural transformation $epsilon: F compose G -> id_{cal(M)}$ that satisfy the following conditions:
  1. For each world model $W in cal(W)$, the morphism $eta_W: W -> G(F(W))$ is a morphism in $cal(W)$.
  2. For each MDP $M in cal(M)$, the morphism $epsilon_M: F(G(M)) -> M$ is a morphism in $cal(M)$.
  3. The following triangle identities hold:
     + For each world model $W in cal(W)$, we have $epsilon_{F(W)} compose F(eta_W) = id_{F(W)}$.
     + For each MDP $M in cal(M)$, we have $G(epsilon_M) compose eta_{G(M)} = id_{G(M)}$.
]

We can measure the accuracy of the world model by counit of the adjunction, which provides a way to evaluate how well the world model captures the dynamics of the environment. The unit morphism $eta_W: W -> G(F(W))$ allows us to compare the predictions made by the world model with the actual outcomes in the MDP, providing a quantitative measure of the model's performance. Since the functor $G$ is lossy, the world model is a simpler representation of the MDP, which allows the agent to make decisions easier.  

We can learn this adjunction difference using JEPA, where the loss function can be interpreted the counit of the adjunction, and the optimization process can be seen as minimizing this loss to improve the accuracy of the world model. Using the world model, we can maintain the state-reward structure and allow the agent make easier decisions in the environment.

#image("/assets/image.png", width: 200pt)




== Conditional network as memory buffer

We train a network to predict the encoding of the environment state, which used to consolidate information stored in memory buffer. 

The agent neural networks make decisions using the memory buffer prediction network as the conditional network. 

