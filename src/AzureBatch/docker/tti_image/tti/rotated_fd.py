# Import modules
from devito import Function, left, right
from devito.finite_differences import centered, first_derivative, transpose, left, right, div, grad
from devito.symbolics import retrieve_functions

def Gzz(field, costheta, sintheta, cosphi, sinphi, rho):
    """
    3D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    order1 = field.space_order // 2
    func = list(retrieve_functions(field))[0]
    if func.grid.dim == 2:
        return Gzz2d(field, costheta, sintheta, space_order)
    x, y, z = func.space_dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x, side=centered,
                                                fd_order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y, side=centered,
                                                fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered,
                                       fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * cosphi * rho,
                            dim=x, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi * rho,
                            dim=y, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta * rho,
                            dim=z, side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


def Gzz2d(field, costheta, sintheta, rho):
    """
    3D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt ztranspose
    """
    order1 = field.space_order // 2
    func = list(retrieve_functions(field))[0]
    x, z = func.space_dimensions
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * rho, dim=x, side=centered,
                            fd_order=order1, matvec=transpose) +
           first_derivative(Gz * costheta * rho, dim=z, side=centered,
                            fd_order=order1, matvec=transpose))
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy(field, costheta, sintheta, cosphi, sinphi, rho):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: Sum of the 3D rotated second order derivative in the direction x and y
    """
    lap = laplacian(field, rho)
    func = list(retrieve_functions(field))[0]
    if func.grid.dim == 2:
        Gzzr = Gzz2d(field, costheta, sintheta, rho)
    else:
        Gzzr = Gzz(field, costheta, sintheta, cosphi, sinphi, rho)
    return lap - Gzzr


def laplacian(v, rho):
    if rho is None:
        Lap = v.laplace
        rho = 1
    else:
        if isinstance(rho, Function):
            Lap = grad(rho).T * grad(v) + rho * v.laplace
        else:
            Lap = rho * v.laplace
    return Lap

