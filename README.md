# Creative Social Robot #

This repository contains 
- A prototypical python implementation of an architecture for 
creative long-term personalization of Socially Assistive Robots (SARs) combining fuzzy logic control and genetic algorithms. 
The protoype allows a SAR to autonomously evolve, via genetic algorithms, the fuzzy rules that govern its interactions, 
and in particular the types of activities that the SAR can suggest and conduct together, with the patient. 
The evolutionary process allows to perform long-term personalization in a creative way.
The evolution of the rules makes use of a fitness function that accounts for 
  - the experience acquired through the interactions with the patient, encoded as credits associated to the rules,
  - the indications given by the therapists, expressed via a second fuzzy rule base (the assessor rule base), 
  - A diversity index (which combines two Gini-Simpson indeces) to guarantee variety in the rule base
- An agent-based simulation used to evaluate the proposal by simulating interactions between the SAR and 
synthetic patients randomly generated. The simulation involves 3 types of agents: 
  - the SAR agent
  - the Patient agent, simulating a simple patient with its own preferences about different activities 
  and reacting to repeated suggestions of the same activities in different ways. 
  It is possible to run different simulations with different randomly generated patients, each of them with different preferences and ways of react to repeated activities 
  - the Environment agent, randomly changing at every step the value of input contextual variables describing the context of the interaction

### How do I get set up? ###
The repository heavily depends on three libraries: the ```SkFuzzy```, ```PyGAD```, and ```MESA``` libraries.
Full dependencies requirements can be found in the file requirements.txt

After installing the required dependencies,
to run experiments it is sufficient to set the parameters in the run_experiments.py file and run the same file.

In order to use the SAR prototype in a different environment form the simulation one reported here, 
it is required to extract the folders
- ```simulation/agents/sar```
- ```simulation/agents/utils```
and adapt the sar_agent.py file to support the new environment
(here the SAR behavior at every simulation step is determined in the function ```step```)

#### Repository structure ####
```
Creative Social Robots
│   README.md                               
│   requirements.txt                            # python dependencies
│   run_experiments.py                          # Module to run experiments
│   run_expeirment_commandline.py               # Module to run single experiment via commandline
│   run_experiments_hpc_batch_chain.py          # Module to run experiments on a HPC with PBS
│
└───results                                     # Results reported in the paper
│   └───noise                                   # Results obtained for S in the noise experiments
│   └───syntheticpatients                       # Results obtained for the random patients without noise 
│       └───random
│           └───baseline                        # S_B
│           └───nodiv                           # S_D
│           └───norepcost                       # S_R
│           └───star                            # S
│   
└───simulation                                  # The modules necessary to run a simulation
│   |   interactionmodel.py                     # MAS Model (in MESA library terminology), to schedule and execute agents
│   |   run_simulation.py                       # Module to run the simulation
│   └───agents                                  # The modules of the different agents
│   └───data                                    # The modules with the necessary input data for the agents
│   └───utils                                   # Utilities for the simulation
│   
└───utils                                       # Utilities for the experiments


```

### Who do I talk to? ###

* Dr. D. Dell'Anna [d.dellanna@tudelft](mailto:d.dellanna@tudelft.nl)
* Dr. A. Jamshidnejad
