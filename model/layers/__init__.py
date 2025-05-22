from .GNNLayer import GNNLayer, SimpleGNNLayer
from .TetraTNNLayer import TetraTNNLayer
from .ClusterTNNLayer import ClusterTNNLayer
from .TNNLayer import TNNLayer


'''    
Notes
-----
- We modify the Higher-order Attention Network (HOAN) from [H23]. 
- Attention mechanisms are removed, and intra-neighborhood aggregation methods are added.
- Cell-Cell E(3)-Invariants (CCI) are added for use.
- This module serve as a base layer for all the TNN layers.

References
----------
.. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
    Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
    (2023) https://arxiv.org/abs/2206.00606.

.. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
    Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
    (2023) https://arxiv.org/abs/2304.10031.

.. [TopoModelX] https://github.com/pyt-team/TopoModelX/blob/main/topomodelx/nn/combinatorial/

'''