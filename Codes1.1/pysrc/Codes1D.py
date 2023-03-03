import jax
import jax.numpy as jnp
from typing import NamedTuple
import sys


def JacobiGQ(alpha,beta,N):

    """JacobiGQ(alpha,beta,N)
        Compute the N'th order Gauss quadrature points, x, 
        and weights, w, associated with the Jacobi 
        polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    """

    if N == 0:
        x = -(alpha-beta)/(alpha+beta+2)
        w = 2
        return x, w

    # Form symmetric matrix from recurrence.
    J = jnp.zeros((N+1))
    h1 = 2*jnp.arange(0,N+1)+alpha+beta
    J = jnp.diag(-1/2*(alpha^2-beta^2)/(h1+2)/h1) + jnp.diag(2/(h1[0:N]+2)*
        jnp.sqrt(jnp.arange(1,N+1)*(jnp.arange(1,N+1)+alpha+beta)*
        (jnp.arange(1,N+1)+alpha)*(jnp.arange(1,N+1)+beta)/(h1[0:N]+1)/(h1[0:N]+3)),1)
    eps = sys.float_info.epsilon
    if alpha+beta<10*eps:
        J = J.at[1,1].set(0)
    J = J + J.T

    #Compute quadrature by eigenvalue solve
    [V,D] = jnp.linalg.eig(J); x = jnp.diag(D)
    # w = (V(1,:)').^2*2^(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*...
    #     gamma(beta+1)/gamma(alpha+beta+1);
    return

def JacobiGL(alpha,beta,N):
    """Compute the N'th order Gauss Lobatto quadrature 
        points, x, associated with the Jacobi polynomial,
        of type (alpha,beta) > -1 ( != -0.5).
    """ 

    x = jnp.zeros(N+1,1)
    if (N==1):
        x = jnp.array([-1,1])
        return x

    xint, _ = JacobiGQ(alpha+1,beta+1,N-2);
    x = jnp.pad(xint, (1,1), constant_values=(-1,1));
    return x

