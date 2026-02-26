# Research
This folder is intended for research purposes which goes beyond the MISO application. The reason this is located in this repository is that many of the experiments will be run using MISO architectures, data loaders, utilities etc. 

# Multiresolution Hash-Encoding (MHE)

### Representation Definitions

MHE aims to learn a function: $$f(\mathbf{x})=D_\theta(\mathbf{y})$$Where $\theta$ denotes trainable parameters and $\mathbf{y}$ is a trainable encoding: $$\mathbf{y}=\mathbf{e}(\mathbf{x};\Phi)$$With the trainable parameters $\Phi$. In which the encoding is arranged into $L$ levels, each containing up to $T$ feature vectors of dimension $F$. For each layer $l \in L$, we can calculate spatial indexes for a query point $\mathbf{x} \in \mathbb{R}^3$: $$\begin{align}N_l &:=\lfloor N_\text{min} \cdot b^l \rfloor \\ \mathbf{x}_l &:= \mathbf{x} \cdot N_l \\ \mathcal{V}(\mathbf{x}_l) &:= \set{\lfloor \mathbf{x}_l \rfloor, \lceil \mathbf{x}_l \rceil}^3: \mathbb{R}^3 \rightarrow \mathbb{Z}^{3 \times 8} \end{align}$$Where $\mathcal{V}(\mathbf{x}_l)$ spans out a cube with $2^3=8$ vertex indices. Each of the vertex indices $\mathbf{v}_i \in \mathcal{V}(\mathbf{x}_l)$ in the cube that $\mathbf{x}_l$ falls within are hashed from the unbounded spatial set of integers $\mathbf{v}_i \in \mathbb{Z}^3$ to a set of indices in the bounded hash table $T \in \mathbb{Z}_T$ using the spatial hash function: $$h(\mathbf{v}_i) = \Big( \bigoplus_{i=1}^3 v_{i} \pi_i \Big) \ \text{mod}\ T \ : \mathbb{Z}^3 \rightarrow \mathbb{Z}_T$$Where $\oplus$ denotes the bit-wise XOR operation using big primes to create pseudo-random permutations. $$\begin{align}\pi_1 &= 1 \\ \pi_2 &= 2\ 654\ 435\ 761 \\ \pi_3 &= 805\ 459\ 861 \\ \end{align}$$
The feature vector for level $l$, denoted as $\mathbf{f}_l(\mathbf{x})$, is computed by **trilinear interpolation** of the feature vectors at the $2^3 = 8$ vertices $\mathbf{v}_i \in \mathcal{V}(\mathbf{x}_l)$. This is mathematically defined as a weighted sum:

$$\mathbf{f}_l(\mathbf{x}) = \sum_{\mathbf{v}_i \in \mathcal{V}(\mathbf{x}_l)} w(\mathbf{x}_l, \mathbf{v}_i) \cdot T(n_{l,i})$$
Where $T(n_{l,i}) \in \mathbb{R}^F$ is the feature vector retrieved from the $l$-th hash table. The weight $w(\mathbf{x}_l, \mathbf{v}_i)$ is determined by the **interpolation kernel** $K(\mathbf{u})$, defined as a product of 1D tent functions:

$$w(\mathbf{x}_l, \mathbf{v}_i) = K(\mathbf{x}_l - \mathbf{v}_i) = \prod_{d=1}^3 \max(0, 1 - |x_{l,d} - v_{i,d}|)$$

This kernel ensures that the contribution of each vertex is proportional to its proximity to the query point $\mathbf{x}_l$. By the properties of the tent function, the weights for all 8 vertices sum to unity: $\sum_{i=1}^8 w(\mathbf{x}_l, \mathbf{v}_i) = 1$.

We then concatenate all the interpolated features produced by all the layers into a single latent vector to get our final encoding: $$\mathbf{e}(\mathbf{x};\theta)= [\mathbf{f}_1(\mathbf{x})^\top, \dots, \mathbf{f}_L(\mathbf{x})^\top]^\top \in \mathbb{R}^{L \times F}$$We then train a decoder $m(\mathbf{y};\Phi)$ that takes in the encoded input and predicts our target value.

### Definitions:

- $\Phi$: Trainable decoder parameters.
- $\theta$: Trainable encoder parameters.
- $l \in L$: Layer index in interval $[1,L]$.
- $N_l$: Resolution in number of cells at layer $l$.
- $\mathbf{x} \in \mathbb{R}^3$: 3D query position vector.
- $b$: Scale factor of resolution between each layer.
- $T$: Hash table size.
- $h(\mathbf{x}_l)$: Hash function that maps spatial indices at layer $l$ to hash table indices.


---
### Collision Metric

I want to quantify how much the hash collisions impact the performance through a ***spatial collision density analysis***. There are multiple ways of quantifying the collision effects. Firstly we can look at the frequency of collisions, ignoring the size of the gradients. The two ways to quantify this is to calculate the ***potential collisions***: $$C_{pot}(l,i) = | \set{\mathbf{v} \in \mathcal{S}_l : h(\mathbf{v}=i)}|$$Where $\mathcal{S}_l$ is the set of all possible voxel vertices in a scene in layer $l$. This will give us a collision map for all the potential number of collisions that can occur for the given voxels. This is agnostic to the training data we input, and works as a theoretical upper bound for any given scene $\mathcal{S}$.

However, during training we're not guaranteed to sample from every voxel in our grid, and a more representative measurement of collisions for a specific scene is therefore to take a data driven approach. We can define the set of "active" vertices $\mathcal{V}_{\text{active}}$ as the set of all integer vertices accessed during training. By counting only the collisions the active voxels experience, we calculate the ***effective collisions***:$$C_{eff}(l,i) = | \set{\mathbf{v} \in \mathcal{V}_{\text{active},l} : h(\mathbf{v}=i)}|$$The InstantNGP paper states that hash collisions are usually resolved because the largest gradient will dominate the regions with many collisions. This implies that a "bad" region is not just one with many collisions, but one where the colliding points have high importance. This leads us to define a metric that quantifies the "conflict" at index $i$ by measuring the **total gradient magnitude** accumulated into that bin during training:$$C_{grad}(l,i) = \sum_{j \in \mathcal{X}_i} \| \nabla_{\theta_{l,i}} \mathcal{L}(\mathbf{x}_j) \|$$where $\mathcal{X}_i$ is the set of all training samples $\mathbf{x}_j$ whose surrounding voxel vertices map to the hash index $i$.
