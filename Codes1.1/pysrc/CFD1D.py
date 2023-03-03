import jax
import jax.numpy as jnp

from globals1D import *

gamma = 1.4

def EulerRHS1D(rho, rhou, Ener):
  """Evalueate RHS flux for 1D Euler Equations
  """
  # compute max velocity for LF Flux
  pres = (gamma-1)*(Ener - 0.5*(rhou)**2 / rho)
  cvel = jnp.sqrt(gamma*pres / rho)
  lm = jnp.abs(rhou / rho) + cvel

  # compute flux
  rhof = rhou
  rhouf = rhou**2 / rho + pres
  Enerf = (Ener+pres)*rhou / rho

  # compute jumps
  drho = rho[vmapM] - rho[vmapP]
  drhou = rhou[vmapM] - rhou[vmapP]
  dEner = Ener[vmapM] - Ener[vmapP]
  drhof = rhof[vmapM] - rhof[vmapP]
  drhouf = rhouf[vmapM] - rhouf[vmapP]
  dEnerf = Enerf[vmapM] - Enerf[vmapP]
  LFc = jnp.max(lm[vmapM], lm[vmapP])

  # compute flux at interfaces
  drhof = nx*drhof/2 - LFc/2*drho
  drhouf = nx*drhouf/2 - LFc/2*drhou
  dEnerf = nx*dEnerf/2 - LFc/2*dEner

  # BC's for shock tube
  rhoin = rho[0]
  rhouin = 0
  pin = pres[0]
  Enerin = Ener[-1]
  rhoout = rho[-1]
  rhouout = 0
  pout = pres[-1]
  Enerout = Ener[-1]

  # set fluxes at inflow/outflow
  rhofin = rhouin
  rhoufin = rhouin**2 / rhoin + pin
  Enerfin = (pin / (gamma-1) + 0.5*rhouin**2 / rhoin + pin)*rhouin / rhoin
  lmI = lm[vmapI] / 2
  nxI = nx[vmapI]
  drho = drho.at[mapI].set(nxI @ (rhof[vmapI] - rhofin) / 2 -
                           lmI @ (rho[vmapI] - rhoin))
  drhou = drhou.at[mapI].set(nxI @ (rhouf[vmapI] - rhoufin) / 2 -
                             lmI @ (rhou[vmapI] - rhouin))
  dEner = dEner.at[mapI].set(nxI @ (Enerf[vmapI] - Enerfin) / 2 -
                             lmI @ (Ener[vmapI] - Enerin))

  rhofout = rhouout
  rhoufout = rhouout**2 / rhoout + pout
  Enerfout = (pout /
              (gamma-1) + 0.5*rhouout**2 / rhoout + pout)*rhouout / rhoout
  lmO = lm[vmapO] / 2
  nxO = nx[mapO]
  drho = drho.at[mapO].set(nxO @ (rhof[vmapO] - rhofout) / 2 -
                           lmO @ (rho[vmapO] - rhoout))
  drhou = drhou.at[mapO].set(nxO @ (rhouf[vmapO] - rhoufout) / 2 -
                             lmO @ (rhou[vmapO] - rhouout))
  dEner = dEner.at[mapO].set(nxO @ (Enerf[vmapO] - Enerfout) / 2 -
                             lmO @ (Ener[vmapO] - Enerout))

  # compute rhs of the PDEs
  rhsrho = -rx*(Dr@rhof) + LIFT @ (Fscale*drhof)
  rhsrhou = -rx*(Dr@rhouf) + LIFT @ (Fscale*drhouf)
  rhsEner = -rx*(Dr@Enerf) + LIFT @ (Fscale*dEnerf)
  return rhsrho, rhsrhou, rhsEner