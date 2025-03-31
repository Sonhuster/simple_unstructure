This code covers generating geometry, unstructured mesh (Tri and Quad element types) and solving the 2D Navier-Stokes equation following the course "FVM for unstructured mesh" from Prof. Sandip Mazumder (https://www.youtube.com/playlist?list=PLVuuXJfoPgT5xu9on_elIP67XblmQ5QBC).
For now, Lid-driven cavity case was set. More cases are going to be setup...
Solver:
  - SIMPLE.
  - 1st order upwind spatial scheme for convective terms.
  - 2nd order scheme for diffusion terms.
  - PWIM scheme for flux calculations.

Programing language : Python.

Package:
  - Numpy.
  - Scipy.spatial Delaunay.
