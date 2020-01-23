from forward import *
from gradient import *
from born import *

from devito.tools import memoized_meth
from devito import TimeFunction


class TTIPropagators(object):
    """
    Propagators in a tti media including mine, self-adjoint, time-reversed,..

    Forward, adjoint and gradient
    """

    def __init__(self, model, space_order=12):

        self.model = model
        self.space_order = space_order

    @memoized_meth
    def op_fwd(self, save=False, sub=None, norec=False, fs=False):
        return forward(self.model, save=save, space_order=self.space_order,
                       sub=sub, norec=norec, fs=fs)

    @memoized_meth
    def op_grad(self, save=False, sub=None, fs=False, isic=False):
        return gradient(self.model, save=save, space_order=self.space_order,
                        sub=sub, fs=fs, isic=isic)

    @memoized_meth
    def op_born(self, save=False, sub=None, recu=False, fs=False, isic=False):
        return born(self.model, save=save, space_order=self.space_order,
                    sub=sub, recu=recu, fs=fs, isic=isic)

#########################################################################################
    def forward(self, src, rec_coordinates, save=False, model=None, norec=False, **kwargs):

        model = model or self.model
        kwargs.update(model.physical_params())

        kwargs["src"] = src
        nt = src.time_range.num

        if not norec:
            rec = Receiver(name='rec', grid=model.grid,
                           time_range=src.time_range,
                           coordinates=rec_coordinates)
            kwargs["rec"] = rec
        else:
            rec = 0

        fs = kwargs.pop('freesurface', False)
        # Subsampling
        sub = kwargs.pop('sub', None)
        if sub is not None and (sub[0] > 1 or sub[1] > 1):
            usave, vsave = subsampled(model, nt, self.space_order,
                                        t_sub=sub[0], space_sub=sub[1])
            u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=self.space_order)
            v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=self.space_order)
            self.op_fwd(save=save, sub=sub, norec=norec, fs=fs).apply(u=u, v=v, us=usave,
                                                                      vs=vsave, **kwargs)
        else:
            usave = vsave = None
            u = TimeFunction(name='u', grid=model.grid, save=nt if save else None,
                             time_order=2, space_order=self.space_order)
            v = TimeFunction(name='v', grid=model.grid, save=nt if save else None,
                             time_order=2, space_order=self.space_order)
            self.op_fwd(save=save, norec=norec, fs=fs).apply(u=u, v=v, **kwargs)

        return rec, usave or u, vsave or v

#########################################################################################
    def gradient(self, adj_src, u, v, save=False, model=None, grad=None,
                 sub=None, **kwargs):

        model = model or self.model

        p = TimeFunction(name='p', grid=model.grid, time_order=2, space_order=self.space_order)
        q = TimeFunction(name='q', grid=model.grid, time_order=2, space_order=self.space_order)
        grad = grad or Function(name="grad", grid=u.grid, space_order=0)

        kwargs.update(model.physical_params())

        fs = kwargs.pop('freesurface', False)
        isic = kwargs.pop('isic', False)
        op_kwargs = {'save': save, 'sub':sub, 'fs':fs, 'isic':isic}
        if fs:
            kwargs['fs_m'] = model.nbpml - self.space_order
        if sub is not None and (sub[0] > 1 or sub[1] > 1):
            summary = self.op_grad(**op_kwargs).apply(p=p, q=q, us=u, vs=v, grad=grad,
                                            src=adj_src, **kwargs)
        else:
            summary = self.op_grad(**op_kwargs).apply(p=p, q=q, u=u, v=v, grad=grad,
                                            src=adj_src, **kwargs)
        return grad, summary

#########################################################################################
    def born(self, src, rec_coordinates, save=False, model=None, recu=False, **kwargs):

        model = model or self.model
        kwargs.update(model.physical_params(is_born=True))

        kwargs["src"] = src
        nt = src.time_range.num
        rec = Receiver(name='rec', grid=model.grid,
                       time_range=src.time_range,
                       coordinates=rec_coordinates)
        kwargs["rec"] = rec
        if recu:
            recu = Receiver(name='recu', grid=model.grid,
                           time_range=src.time_range,
                           coordinates=rec_coordinates)
            kwargs["recu"] = recu
        else:
            recu = None

        ul = TimeFunction(name='ul', grid=model.grid, time_order=2, space_order=self.space_order)
        vl = TimeFunction(name='vl', grid=model.grid, time_order=2, space_order=self.space_order)

        fs = kwargs.pop('freesurface', False)
        isic = kwargs.pop('isic', False)
        sub = kwargs.pop('sub', None)
        op_kwargs = {'save': save, 'sub': sub, 'recu': recu, 'fs':fs, 'isic':isic}
        if sub is not None and (sub[0] > 1 or sub[1] > 1):
            usave, vsave = subsampled(model, nt, self.space_order, t_sub=sub[0], space_sub=sub[1])
            u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=self.space_order)
            v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=self.space_order)
            summary = self.op_born(**op_kwargs).apply(u=u, v=v, us=usave,vs=vsave, ul=ul, vl=vl, **kwargs)
        else:
            usave = vsave = None
            u = TimeFunction(name='u', grid=model.grid, save=nt if save else None,
                             time_order=2, space_order=self.space_order)
            v = TimeFunction(name='v', grid=model.grid, save=nt if save else None,
                             time_order=2, space_order=self.space_order)
            summary = self.op_born(**op_kwargs).apply(u=u, v=v, ul=ul, vl=vl, **kwargs)

        if save:
            return rec, usave or u, vsave or v, summary
        else:
            return rec, recu, summary

