# Crystal-growth
This software is a lattice–based simulator for the growth of crystalline structures under different kinetic regimes, inspired by this paper[^1]. 
It implements both an EDEN type model (interface–controlled growth) and a diffusion–limited aggregation (DLA) model (transport–controlled growth), on 2D and 3D cubic lattices.  
It expands the work in the paper by adding new features that allow to simulate a bigger set of physical phenomena.  
The software allows both a quantitative analysis, by computing relevant properties, and qualitative analysis, by plotting the final cristal, so that the user can directly see its final structure.



## Introduction
Crystals are solids where the building units, called *monomers* sit in a periodically ordered structure, repeating in space like a tiling. This long–range order is what gives crystals well–defined faces, sharp melting points, and very directional physical properties.

The main physical phenomena in which these objects are involved are:
* **Nucleation** - the birth of a new crystal from a melt, solution, or vapor: small clusters form, most die, a few exceed a critical size and become stable nuclei.
* **Growth** - atoms or molecules attach to the crystal surface, often controlled by diffusion (how fast stuff arrives) and interface kinetics (how fast it actually sticks and reorganizes).
* **Anisotropy** - because the lattice is not the same in every direction, properties like growth rate, surface energy, conductivity, optical index, etc. depend on orientation.
* **Defects** - vacancies, dislocations, impurities and grain boundaries break perfect order; they control mechanical strength, plasticity, transport, and often where and how growth proceeds.
* **Phase transitions** - crystals can melt, transform into other crystalline phases, or undergo order–disorder transitions when temperature, pressure, or composition change.

### Lattice representation
The system is defined on a discrete cubic lattice

$$\Lambda = \\{(x,y,z) | x=0,\dots,N_x-1;\\; y=0,\dots,N_y-1;\\; z=0,\dots,N_z-1\\}$$

Each site carries:
* a **binary occupancy** variable $n(\mathbf{r}) \in \\{0,1\\}, \quad \mathbf{r} = (x,y,z)$  
where $n(\mathbf{r}) = 1$ denotates an occupied site and $n(\mathbf{r})=0$ denotates an empty cell.
* an **hisotry** field $h(\mathbf{r}) \in \\{-1,1,2,\dots \\}$  
where $h(\mathbf{r})=-1$ for empty sites, and $h(\mathbf{r})=t$ if the site was occupied at time step $t$.
* an optional **crystal seed set** $S_0 \subset  \Lambda$, used to define the initial occupied regions.

Nearest-neighbor connectivity is defined in the usual 6-neighbors sense in 3D (restricted accordingly in 2D):

$$N(\mathbf{r}) = \\{ \mathbf{r} \pm \hat{x}, \mathbf{r} \pm \hat{y}, \mathbf{r} \pm \hat{z} \\}$$

The **active border** $B(t)$ of the crystal at epoch $t$ is defined as the set of empty sites that are nearest neighbors of at least one occupied sites:

$$B(t) = \\{ \mathbf{r} \in \Lambda | n(\mathbf{r}) = 0, \\; \exists \mathbf{r'} \in N(\mathbf{r}) \text{ with } n(\mathbf{r'})=1 \\}$$

### EDEN growth model
The Eden model implemented here describes growth controlled primarily by the interface kinetics: the crystal expands by occupying sites on its active border.

At each epoch $t$:
1. Compute the active border $B(t)$.
2. Select a site $\mathbf{r} \in B(t)$ with a probability $P(\mathbf{r})$.
3. Set $n(\mathbf{r})=1$ and $h(\mathbf{r}) = t$.

In the isotropic case, all border sites are equivalent, having each a probability:

$$P(\mathbf{r}) = \frac{1}{|B(t)|}$$

With anisotropy enabled, $P(\mathbf{r})$ is biased using a lattice-stored anisotropy weight $w(\mathbf{r})$, so that

$$P(\mathbf{r}) = \frac{w(\mathbf{r})}{\sum_{\mathbf{r'} \in B(t)} w(\mathbf{r'})}$$

The simulator supports both fully 3D and 2D EDEN growth, with the 2D version having the $z$-plane fixed..

### DLA growyh model
The DLA model describes growth limited by particle transport: particles diffuse in the empty region via random walk, and irreversibly attach to the crystal when they reach its neighborhood.

For each new particle in the simulation:
1. Define a **generation bounding box** around the crystal, with padding $p_{gen}$.  
   A random starting position $mathbf{r}_0$ is selected on the surface of this box.
2. Define a **outer bounding box** with larger padding $p_{out} > p_{gen}$.  
   If the particle exits this region or exceeds a maximum number of steps, its walk is restarted from a new random generation point.
3. From $\mathbf{r}\_0$, perform a random walk: $\mathbf{r}_{t+1} = \mathbf{r}_t + \mathbf{\Delta r}_t$  
   where $\mathbf{\Delta r}_t$ is the nearest-neighbor step, chosen according to a probability distribution that can be isotropic or not.
4. If at a given step the particle position $\mathbf{r}_t$ has at least one occupied neighbor, the particle sticks to the crystal. The walk ends and the simulator proceeds with the next particle.

The simulator collects simple statistics on the random walk, such as mean number of steps and mean number of restarts per attached particle.

### Directional anisotropy
The software provides a general anisotropy mechanism implemented at the lattice level. 
Anisotropy is controlled by:
* a set of preferred **growth directions** $\\{ \mathbf{\hat{a}_i} \\}$, each being a unit vecotr in $\mathcal{R}^3$.
* a scalar **anisotropy strength** $\kappa \ge 0$.

For any direction $\mathbf{v} \ne 0$ (e.g. a border–site position relative to a reference point, or a random–walk step), the lattice computes an **anisotropy weight**:

$$w(\mathbf{r}) = \sum_i e^{\kappa \cdot cos\theta_i}, \quad cos\theta_i = \frac{\mathbf{v \cdot \hat{a}_i}}{|\left| \mathbf{v} \right||}$$

In the EDEN model, $\mathbf{v}$ is typically taken as the vector from a fixed reference (e.g. the mean seed position) to the candidate border site, biasing the selection of growth sites.  
In the DLA model, $\mathbf{v}$ is taken as the candidate random–walk step $\mathbf{\Delta r}$; the probability of choosing a step from the set of nearest–neighbor directions $\\{\mathbf{\Delta r}_j\\}$ becomes:

$$P(\mathbf{\Delta r}_j) = \frac{w(\mathbf{\Delta r}_j)}{\sum_k w(\mathbf{\Delta r}_k)}$$

By choosing appropriate sets of $\\{ \mathbf{\hat{a}_i} \\}$ (e.g. three directions at 120° in 2D) and tuning $\kappa$, the user can generate crystals with a controlled number of preferred growth arms or with strongly biased growth along specific axes.

When anisotropy is disabled (no directions or $\kappa=0$), all weights reduce to $w=1$, and the models revert to their standard isotropic versions.

### Analysis tools
The software includes basic analysis utilities, including:
* **Fractal dimention estimation** of DLA clusters via box-counting over multiple length scales: if $N(l)$ is the number of occupied boxes of side $l$, the dimension $D_f$ is estimated from:

$$N(l) \approx l^{-D_f}$$

* **Distance from active surface analysis**, used in a built-in simulation, quantifying how far occupied sites are from the initial growth plane as a function of epoch.

These tools are intended to provide quick quantitative diagnostics of the generated morphologies.


## Repository Structure


## Getting Started
This software is tested working in the python 3.10 version.

1. Clone the repository: inside the target directory, clone this repo by writing the following in the command prompt
``` 
git clone https://github.com/MatteoAgu12/Crystal-growth
``` 
If the repository has been successfully installed, you should be able to see it after entering the following command in the terminal
``` 
ls
``` 

2. Move to the software directory with
``` 
cd ./Crystal-growth/
``` 
and install all the required dependencies by executing (*pip* version, use your enviroment otherwise)
``` 
pip install -r requirements.txt
```

By doing these few steps you should be ready to use the software on your machine.



## Tutorials
The usage of this software is very easy, and to help the user to learn the main features the following short tutorials are provided, covering all the possibilities offered by this work.

### Create a $\texttt{Lattice}$ object

### Run a built-in simulation

### Create your custom simulation



## Examples
### 2D isotropic DLA simulation

### 2D anisotropic DLA simulation

### Active surface simulation

### Policrystal simulation



## References
[^1]: T. A. Witten & L. M. Sander, *Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon*, Phys. Rev. Lett. 47, 1400–1403 (1981).
