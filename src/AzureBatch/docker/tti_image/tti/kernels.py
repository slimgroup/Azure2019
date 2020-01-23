from sympy import sqrt
from rotated_fd import *


def kernel_zhang_fwd(u, v, costheta, sintheta, cosphi, sinphi,
                     epsilon, delta, rho):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle, has to be 0 in 2D
    :param sinphi: sine of the azymuth angle, has to be 0 in 2D
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 2D
    """
    Gxx = Gxxyy(u, costheta, sintheta, cosphi, sinphi, rho)
    if u.grid.dim == 2:
        Gzzr = Gzz2d(v, costheta, sintheta, rho)
    else:
        Gzzr = Gzz(v, costheta, sintheta, cosphi, sinphi, rho)
    return ((1 + 2 * epsilon) * Gxx + sqrt(1 + 2 * delta) * Gzzr,
            sqrt(1 + 2 * delta) * Gxx + Gzzr)
