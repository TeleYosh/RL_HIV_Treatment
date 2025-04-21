## Project Description

I trained a DQN model to learn which traitment to administer to an HIV patient. The environnement is a simulation of the patient's immune system. 

`train.py`: Contains the architechture of the neural network and the training. It uses a replay buffer and a target network.

## Repository Structure

The repository is organized as follows:
```
├── README.md 
├── requirement.txt # Lists all the necessary dependencies.
└── src
    ├── env_hiv.py # Defines the simulation environment the agent will interact with. 
    ├── evaluate.py # Contains the evaluation function to assess the agent's performance. 
    ├── interface.py # Provides the agent interface, outlining the methods the agent must implement. 
    ├── train.py # Training script. 
    └── main.py 
```

## Environnement 

The `HIVPatient` class implements a simulator of the patient's immune system through 6 state variables, which are observed every 5 days (one time step):
- `T1` number of healthy type 1 cells (CD4+ T-lymphocytes), 
- `T1star` number of infected type 1 cells,
- `T2` number of healthy type 2 cells (macrophages),
- `T2star` number of infected type 2 cells,
- `V` number of free virus particles,
- `E` number of HIV-specific cytotoxic cells (CD8 T-lymphocytes).

The physician can prescribe two types of drugs:
- Reverse transcriptase inhibitors, which prevent the virus from modifying an infected cell's DNA,
- Protease inhibitors, which prevent the cell replication mechanism.

Giving these drugs systematically is not desirable. First, it prevents the immune system from naturally fighting the infection. Second, it might induce drug resistance by the infection. Third, it causes many pharmaceutical side effects which may lead the patient to abandon treatment.  
There are four possible choices for the physician at each time step: prescribe nothing, one drug, or both.

The reward model encourages high values of `E` and low values of `V`, while penalizing giving drugs.

The patient's immune system is simulated via a system of deterministic non-linear equations.

By default, the `HIVPatient` class' constructor instantiates always the same patient (the one whose immune system was clinically identified by Adams et al.). However, calling `HIVPatient(domain_randomization=True)` draws a random patient uniformly.

