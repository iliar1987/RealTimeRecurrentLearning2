# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:52:12 2015

@author: Ilia
"""

from matplotlib import pyplot as plt, gridspec
import numpy as np

class _Direction(object):
    horizontal = 'H'
    vertical = 'V'
    transposed = 'T'
    normal = 'N'

class _Plottable(object):
    def __init__(self,value,direction,use_clim=None):
        if ( len(value.shape)==2 ):
            if direction == _Direction.transposed:
                self.value = value.transpose()
            else:
                self.value = np.copy(value)
        else:
            n = value.shape[0]
            if direction == _Direction.horizontal:
                self.value = value.reshape((1,n))
            else:
                self.value = value.reshape((n,1))
        self.use_clim = use_clim
    def MakeIMShow(self,a,clim_override=None,*additional_args,**additional_kwargs):
        if ( not clim_override is None):
            clim = clim_override
        else:
            clim = self.use_clim
        if ( not clim is None ):
            additional_kwargs['clim'] = clim
        I = a.imshow(self.value,interpolation='none',*additional_args,**additional_kwargs)
        return I

class RTRL_Plotter_base(object):
    def __init__(self):
        self.is_active=False
        self.connected_fig_events = []
    def isActive(self):
        return self.is_active
    def RetrieveState(self,rnn):
        pass
    def CreateAxes(self,fig=None):
        if fig is None:
            fig = plt.figure()
#        plt.figure(fig)
        axes_positions = self._GetAxesPositions()
        axeses={}
        for name in axes_positions.iterkeys():
            a = fig.add_subplot(axes_positions[name])
            a.set_title(name)
            a.set_xticks([])
            a.set_yticks([])
            axeses[name] = a
        
        self.axeses = axeses
        self.axes_positions = axes_positions
        self.fig = fig
        self.ConnectCloseEvent(self.OnFigClose)
    def CreateImages(self):
        self.images={}
        for k,a in self.axeses.iteritems():
            I = self.state[k].MakeIMShow(a)
#            I.set_clim(self.weight_clim)
            self.images[k] = I
        self.is_active = True
    def UpdateImages(self,make_redraw=True):
        for k,a in self.axeses.iteritems():
            self.images[k].set_data(self.state[k].value)
        if make_redraw:
            self.fig.canvas.draw_idle()
#        plt.draw()
    def CloseFig(self):
        if ( not self.fig is None ):
            for cid in self.connected_fig_events:
                self.fig.canvas.mpl_disconnect(cid)
            plt.close(self.fig)
            self.is_active = False
    def OnFigClose(self,evt):
        self.is_active = False
        self.fig = None
    def GetFig(self):
        return self.fig
    def ConnectCloseEvent(self,handler):
        if ( not self.fig is None ):
            cid = self.fig.canvas.mpl_connect('close_event', handler)
            self.connected_fig_events.append(cid)


class RTRL_Weight_Plotter(RTRL_Plotter_base):
    def __init__(self,weight_clim=None):
        super(RTRL_Weight_Plotter,self).__init__()
        if weight_clim is None:
            weight_clim = np.array([-1.0,1.0])
        elif isinstance(weight_clim,list):
            weight_clim = np.array(weight_clim)
        self.weight_clim = weight_clim
        self.excitation_clim = np.array([-1.,1.])
    def RetrieveState(self,rnn):
        state = rnn.GetState()
        S = {}
        S['nx'] = state['nx']
        S['nu'] = rnn.nu
        S['ny'] = rnn.ny
        S['x_t'] = _Plottable(rnn.GetCurrentX(),_Direction.vertical,self.excitation_clim)
        S['u_t'] = _Plottable(rnn.GetCurrentU(),_Direction.vertical,self.excitation_clim)
        S['y_tp1'] = _Plottable(rnn.Get_Y_Prediction(),_Direction.horizontal,[0.0,1.0])
        S['x_tp1'] = _Plottable(rnn.GetNextX(),_Direction.vertical,self.excitation_clim)
        S['W_xx'] = _Plottable(state['W_xx'],_Direction.normal,self.weight_clim)
        S['W_xu'] = _Plottable(state['W_xu'],_Direction.normal,self.weight_clim)
        S['W_yx'] = _Plottable(state['W_yx'],_Direction.normal,self.weight_clim)
        S['b_x'] = _Plottable(state['b_x'],_Direction.horizontal,self.weight_clim)
        S['b_y'] = _Plottable(state['b_y'],_Direction.horizontal,self.weight_clim)
        
        self.state = S
    def _GetAxesPositions(self):
        state = self.state
        temp = max([state['nu'],3])
        gs = gridspec.GridSpec(1 + state['nx'] + temp + 2, state['nx']+2+state['ny']*2 + 2)
        
        axes_positions = {}
        axes_positions['W_xx'] = gs[2:(state['nx']+2),
                                            2:(state['nx']+2)]
        axes_positions['W_xu'] = gs[(state['nx']+3):(state['nx']+3+state['nu']),
                                                    2:(state['nx']+2)]
        axes_positions['x_t'] = gs[2:(state['nx']+2),
                                                0]
        axes_positions['b_x'] = gs[0,
                                2:(state['nx']+2)]
        axes_positions['u_t'] = gs[(state['nx']+3):(state['nx']+3+state['nu']),
                                            0]
        axes_positions['x_tp1'] = gs[2:(state['nx']+2),
                                    state['nx']+3]
        axes_positions['W_yx'] = gs[2:(state['nx']+2),
                                (state['nx']+5):(state['nx']+5 + state['ny'])]
        #                        
        axes_positions['b_y'] = gs[(state['nx']+3),
                                       (state['nx']+5):(state['nx']+5 + state['ny'])]
        axes_positions['y_tp1'] = gs[(state['nx']+5),
                                        (state['nx']+5):(state['nx']+5 + state['ny'])]
        return axes_positions
    def CreateImages(self):
        super(RTRL_Weight_Plotter,self).CreateImages()
        self.cbar=plt.colorbar(self.images['W_xx'],ax=self.axeses.values())

from collections import defaultdict

class RTRL_Gradients_Plotter(RTRL_Plotter_base):
    def __init__(self,gradient_clim=None):
        super(RTRL_Gradients_Plotter,self).__init__()
        if isinstance(gradient_clim,list):
            gradient_clim = np.array(gradient_clim)
        self.gradient_clim = gradient_clim
    def RetrieveState(self,rnn):
        S = {}
        S['nx'] = rnn.nx
        S['nu'] = rnn.nu
        S['ny'] = rnn.ny
        weight_gradients = rnn.GetWeightGradients()
#        S.update(weight_gradients)
        directions = defaultdict(lambda : _Direction.normal)
        directions['db_x'] = _Direction.horizontal
        directions['db_y'] = _Direction.horizontal
        for k,v in weight_gradients.iteritems():
            S[k] = _Plottable(v,directions[k],self.gradient_clim)
        self.state = S
        
    def _GetAxesPositions(self):
        state = self.state
        temp = max([state['nu'],3])
        gs = gridspec.GridSpec(1 + state['nx'] + temp + 2, state['nx']+2+state['ny']*2 + 2)
        
        axes_positions = {}
        axes_positions['dW_xx'] = gs[2:(state['nx']+2),
                                            2:(state['nx']+2)]
        axes_positions['dW_xu'] = gs[(state['nx']+3):(state['nx']+3+state['nu']),
                                                    2:(state['nx']+2)]
        axes_positions['db_x'] = gs[0,
                                2:(state['nx']+2)]
        axes_positions['dW_yx'] = gs[2:(state['nx']+2),
                                (state['nx']+5):(state['nx']+5 + state['ny'])]
        #                        
        axes_positions['db_y'] = gs[(state['nx']+3),
                                       (state['nx']+5):(state['nx']+5 + state['ny'])]
        return axes_positions
    def _CalcCommonCLim(self):
        cmin = np.inf
        cmax = -np.inf
        for k in self.images.iterkeys():
            if k == 'db_y':
                continue
            this_max = self.state[k].value.max()
            this_min = self.state[k].value.min()
            if this_max > cmax:
                cmax = this_max
            if this_min < cmin:
                cmin = this_min
        max_abs = max(abs(cmin),abs(cmax))
        return np.array([-max_abs,max_abs],dtype=float)
    def _UpdateAllClims(self):
        for k,I in self.images.iteritems():
            I.set_clim(self.common_clim)
    def _UpdateColorBar(self):
        #self.cbar.
        #probably happens automatically
        pass
    def CreateImages(self):
        super(RTRL_Gradients_Plotter,self).CreateImages()
        if self.gradient_clim is None:
            self.common_clim = self._CalcCommonCLim()
            self._UpdateAllClims()
        self.cbar=plt.colorbar(self.images['dW_xx'],ax=self.axeses.values())
    def UpdateImages(self):
        super(RTRL_Gradients_Plotter,self).UpdateImages(False)
        if self.gradient_clim is None:
            self.common_clim = self._CalcCommonCLim()
            self._UpdateAllClims()
            self._UpdateColorBar()
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
#    RP = RTRL_Weight_Plotter()
    RP = RTRL_Gradients_Plotter()
    from rtrl1 import RTRL1
    rnn1 = RTRL1()
    #rnn1.BuildNetwork()
    
    import cPickle
    with open('well_trained_nx8.rtrl1','rb') as f:
        state = cPickle.load(f)
    rnn1.SetState(state)
    rnn1.BuildNetwork()
    RP.RetrieveState(rnn1)
    RP.CreateAxes()
    RP.CreateImages()

