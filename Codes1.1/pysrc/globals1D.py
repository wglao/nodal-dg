from Codes1D import *

# declare global variables
    
N = 20
    
Np = N+1; Nfp = 1; Nfaces = 2

# Compute basic Legendre Gauss Lobatto grid
r = JacobiGL(0,0,N)

# Build reference element matrices
V  = Vandermonde1D(N, r); invV = inv(V)
Dr = Dmatrix1D(N, r, V);

# Create surface integral terms
LIFT = Lift1D();

# build coordinates of all the nodes
va = EToV(:,1)'; vb = EToV(:,2)';
x = ones(N+1,1)*VX(va) + 0.5*(r+1)*(VX(vb)-VX(va));

# calculate geometric factors
[rx,J] = GeometricFactors1D(x,Dr);

# Compute masks for edge nodes
fmask1 = find( abs(r+1) < NODETOL)'; 
fmask2 = find( abs(r-1) < NODETOL)';
Fmask  = [fmask1;fmask2]';
Fx = x(Fmask(:), :);

# Build surface normals and inverse metric at surface
[nx] = Normals1D();
Fscale = 1./(J(Fmask,:));

# Build connectivity matrix
[EToE, EToF] = Connect1D(EToV);

# Build connectivity maps
[vmapM, vmapP, vmapB, mapB] = BuildMaps1D;

# Low storage Runge-Kutta coefficients
rk4a = jnp.array([
    0.0, -567301805773.0 / 1357537059087.0, -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0, -1275806237668.0 / 842570457699.0
])
rk4b = jnp.array([
    1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0
])
rk4c = jnp.array([
    0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0
])