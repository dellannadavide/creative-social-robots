# Creative Social Robot #

This repository contains 
- A prototypical python implementation of an architecture for 
creative long-term personalization of Socially Assistive Robots (SARs) combining fuzzy logic control and genetic algorithms via EFS4SAR (Evolving Fuzzy system for Socially Assistive Robots). 
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
- Experimental results obtained by both simulating interactions between SAR and randomly synthesized patients, and by using the SAR as a classifier of activities based on actual preferences of humans about daily activities elicited through a survey. In the latter case, the activities suggested by the SAR are evaluated by a fuzzy patient which models the actual preferences of the human and provides feedback about the suggestions. 

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
│───experiments_simulation                      # The modules necessary to run experiments
                                                  with randomly synthesised patients
│   |   run_experiments.py                         # Module to run experiments
│   |   run_expeirment_commandline.py              # Module to run single experiment via commandline
│   |   run_experiments_hpc_batch_chain.py         # Module to run experiments on a HPC with PBS
│
│───experiments_classification                  # The modules necessary to run experiments 
                                                  using the SAR as a classifier of activities 
                                                  trained using data from human preference
│   |   run_experiments_datasets.py                # Module to run experiments
│   |   run_expeirment_ds_commandline.py           # Module to run single experiment via commandline
│   |   run_experiments_ds_hpc_batch_chain.py      # Module to run experiments on a HPC with PBS
│   |   generate_datasets.py                       # Module to generate datasets of activities in different contexts
                                                     (based on data from human preferences stored in 
                                                     '../simulation/data/survey' that will be used by the classifier 
                                                     as train and test set
│   |   eval_sar_on_dataset.py                    # Module to call to evaluate the SAR on one dataset
│   |   eval_classifier.py                        # Module to evaluate the results obtained via classification
│   |   CreativeSocialRobots_Survey.xlsx          # The survey used to elicit human preferences

│
└───results                                     # Results reported in the paper
│   └───syntheticpatients/multiple_measurements # Results obtained for the synthesized patients
│   └───noise                                   # Results obtained for S in the noise experiments
│   └───baseline                                # S_B
│   └───nodiv                                   # S_D
│   └───norepcost                               # S_R
│   └───star                                    # S
│   └───classification                          # Results obtained from EFS4SAR and other 9 EFSs in
                                                  classifying 15 data sets of activities 
│   
└───simulation                                  # The modules necessary to run a simulation
│   |   interactionmodel.py                     # MAS Model (in MESA library terminology), to schedule and execute agents
│   |   run_simulation.py                       # Module to run the simulation
│   └───agents                                  # The modules of the different agents
│   └───data                                    # The necessary input data for the agents
│       └───survey                                # The preferences elicited from 15 human participants of a survey
                                                    and the datasets obtained from such preferences
│   └───utils                                   # Utilities for the simulation
│   
└───utils                                       # Utilities for the experiments


```

### Who do I talk to? ###

* Dr. D. Dell'Anna [d.dellanna@tudelft](mailto:d.dellanna@tudelft.nl)
* Dr. A. Jamshidnejad