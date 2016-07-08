# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:54:52 2015

@author: Ilia

changes from rtrl1: adding a term to the cost which is proportional to deviation of
x_tp1 from 0.5 and deviation of y_tp1 from 0.5
also, am going to change y into single output (cost is binary cross entropy anyway)
"""
import theano
from theano import Param
import theano.tensor as T
import numpy as np

use_deviation_costs = False
def UsingDeviationCosts():
    return use_deviation_costs

def EnableDeviationCosts():
    global use_deviation_costs
    use_deviation_costs=True
    RTRL2.alpha_default_value=0.2
    RTRL2.beta_default_value=0.2

def DisableDeviationCosts():
    global use_deviation_costs
    use_deviation_costs=True
    RTRL2.alpha_default_value=0.0
    RTRL2.beta_default_value=0.0

def jacobian(Y,X):
    J,_ = theano.scan(fn = lambda i,Y,X: T.grad(Y[i],X),
                      sequences = T.arange(Y.shape[0]),
                        non_sequences=[Y,X],
                    outputs_info = None)
    return J

deviation_cost_pow4 = lambda X : 4*X**4

def calc_entropy(X):
    X_s = T.exp(X)
    X_s /= X_s.sum()
    return -(X_s * T.log(X_s)).sum()
    #compute entropy of x understood as probability distribution


def log_deviation_cost(X):
    X_s = T.abs_(X)
    return T.switch(T.ge(X_s,0.99),
             -T.log(1.0-0.99) + 100.0*(X_s-0.99), #tailor around 0.99 to avoid taking log(1.0-1.0)
             -T.log(1.0-X_s))

class RTRL2(object):
#    alpha_default_value =1000.0
    if ( use_deviation_costs):
        alpha_default_value = 0.2
        beta_default_value = 0.2
    else:
        alpha_default_value = 0.0
        beta_default_value = 0.0
    sigma_default = 0.2
    def __init__(self):
        self.Init()
    def Init(self):
        #initialize shared variables to enable SetState and such
        self.nu = 1
        self.ny = 1
        empty_mat = np.matrix(0.0)
        empty_vec = np.array([0.0])
        empty_tensor3 = np.array([[[0.0]]])
        self.W_xu = theano.shared(empty_mat, 'W_xu')
        self.W_xx = theano.shared(empty_mat, 'W_xx')
        self.W_yx = theano.shared(empty_mat, 'W_yx')
        self.b_x = theano.shared(empty_vec, 'b_x')
        self.b_y = theano.shared(empty_vec, 'b_y')
        self.rng = np.random.RandomState()
        self.h = T.tanh
#        self.g = lambda Y: T.flatten(T.nnet.softmax(Y))
        self.g = T.nnet.sigmoid
        
        self.cost_func = lambda Y,V: -(V*T.log(Y) + (1-V)*T.log(1-Y))[0]
#        self.cost_func = lambda Y,V: -((V*0.25+0.5)*T.log(Y) + (1-(V*0.25+0.5))*T.log(1-Y))[0]
        #gave me nans:
#        self.cost_x_deviation = lambda X: deviation_cost(X).mean()
        if ( use_deviation_costs):
            self.cost_x_deviation = lambda X: log_deviation_cost(X).mean()
    #            5*(T.sum(X)-1.0)**6
    #        self.cost_x_deviation = lambda X: ((X*2)**4).max()
    #        self.cost_y_deviation = lambda Y: deviation_cost_pow4(2.0*Y-1.0)[0]
            self.cost_y_deviation = lambda Y: log_deviation_cost(2.0*Y-1.0)[0]
        else:
            self.cost_x_deviation = lambda X: (X*0.0).sum()
            self.cost_y_deviation = lambda Y: (Y*0.0).sum()
        self.x_t = theano.shared(empty_vec,'x_t')
        self.u_t = theano.shared(np.array([-1.0]),'u_t')
        self.dx_t_dW_xu = theano.shared(value=empty_tensor3,name='dx_t_dW_xu')
        self.dx_t_dW_xx = theano.shared(value=empty_tensor3,name='dx_t_dW_xx')
        self.dx_t_dbx = theano.shared(value=empty_mat,name='dx_t_dbx')
        
        self.shared_variables = [self.W_xu,self.W_xx,self.W_yx,self.b_x,self.b_y,self.x_t,self.u_t,self.dx_t_dbx,self.dx_t_dW_xu,self.dx_t_dW_xx]

    def BuildNetwork(self):        
        learning_rate = T.scalar('learning_rate')
        v_tp1 = T.vector('v_tp1')
        alpha = T.scalar('alpha')
        beta = T.scalar('beta')
        
        
        net_x_tp1 = T.dot(self.x_t,self.W_xx) + T.dot(self.u_t,self.W_xu) + self.b_x
        x_tp1 = self.h(net_x_tp1)
        net_y_tp1 = T.dot(self.x_t,self.W_yx) + self.b_y
        y_tp1 = self.g(net_y_tp1)
        
        dx_tp1_dx_t = jacobian(x_tp1,self.x_t)
        dx_tp1_dW_xu = T.tensordot(dx_tp1_dx_t,self.dx_t_dW_xu,[1,0]) + jacobian(x_tp1,self.W_xu)
        dx_tp1_dW_xx = T.tensordot(dx_tp1_dx_t,self.dx_t_dW_xx,[1,0]) + jacobian(x_tp1,self.W_xx)
        dx_tp1_dbx = T.tensordot(dx_tp1_dx_t,self.dx_t_dbx,[1,0]) + jacobian(x_tp1,self.b_x)
        
        dy_tp1_dx_t = jacobian(y_tp1,self.x_t)
        dy_tp1_dW_yx = jacobian(y_tp1,self.W_yx)
        dy_tp1_dby = jacobian(y_tp1,self.b_y)
        dy_tp1_dW_xx = T.tensordot(dy_tp1_dx_t,self.dx_t_dW_xx,[1,0])
        dy_tp1_dW_xu = T.tensordot(dy_tp1_dx_t,self.dx_t_dW_xu,[1,0])
        dy_tp1_dbx = T.tensordot(dy_tp1_dx_t,self.dx_t_dbx,[1,0])
        
        E_tp1 = self.cost_func(y_tp1,v_tp1) \
            + alpha * self.cost_x_deviation(x_tp1) \
            + beta * self.cost_y_deviation(y_tp1)
        
        dE_tp1_dy_tp1 = T.grad(E_tp1,y_tp1)
        dE_tp1_dx_tp1 = T.grad(E_tp1,x_tp1)
        
        dE_tp1_dW_xx = T.tensordot(dE_tp1_dy_tp1,dy_tp1_dW_xx,[0,0]) + \
                        T.tensordot(dE_tp1_dx_tp1,dx_tp1_dW_xx,[0,0])
                        
        dE_tp1_dW_xu = T.tensordot(dE_tp1_dy_tp1,dy_tp1_dW_xu,[0,0]) + \
                        T.tensordot(dE_tp1_dx_tp1,dx_tp1_dW_xu,[0,0])
                        
        dE_tp1_dbx = T.dot(dE_tp1_dy_tp1,dy_tp1_dbx) + \
                        T.dot(dE_tp1_dx_tp1,dx_tp1_dbx)
                        
        dE_tp1_dby = T.dot(dE_tp1_dy_tp1,dy_tp1_dby)
                        
        dE_tp1_dW_yx = T.tensordot(dE_tp1_dy_tp1,dy_tp1_dW_yx,[0,0])
                        
        delta_W_xx = - (learning_rate * dE_tp1_dW_xx)
        delta_W_xu = - (learning_rate * dE_tp1_dW_xu)
        delta_bx = - (learning_rate * dE_tp1_dbx)
        delta_by = - (learning_rate * dE_tp1_dby)
        delta_W_yx = - (learning_rate * dE_tp1_dW_yx)
        
        alpha_param = Param(alpha,default=self.alpha_default_value)
        beta_param = Param(beta,default=self.beta_default_value)

        
        self.get_x_tp1 = theano.function([],x_tp1,allow_input_downcast = True)
        self.get_y_tp1 = theano.function([],y_tp1,allow_input_downcast = True)
        self.get_y_gradients = theano.function([],
                                                    outputs=[dy_tp1_dW_xx,
                                                             dy_tp1_dW_xu,
                                                             dy_tp1_dW_yx,
                                                             dy_tp1_dbx,
                                                             dy_tp1_dby],
                                                    allow_input_downcast = True)
        self.get_cost_gradients = theano.function([v_tp1,alpha_param,beta_param],
                                                    outputs=[dE_tp1_dW_xx,
                                                             dE_tp1_dW_xu,
                                                             dE_tp1_dW_yx,
                                                             dE_tp1_dbx,
                                                             dE_tp1_dby],
                                                    allow_input_downcast = True)
        
        updates_train_weights = [(self.W_xx,self.W_xx+delta_W_xx),
                         (self.W_xu,self.W_xu+delta_W_xu),
                        (self.W_yx,self.W_yx+delta_W_yx),
                        (self.b_x,self.b_x+delta_bx),
                        (self.b_y,self.b_y+delta_by)]
        updates_gradients = [(self.dx_t_dW_xu,dx_tp1_dW_xu),
                                   (self.dx_t_dW_xx,dx_tp1_dW_xx),
                                   (self.dx_t_dbx,dx_tp1_dbx)]
        updates_x = [(self.x_t,x_tp1)]
#        updates_y = [(self.y_t,y_tp1)]
        updates_y = []
        updates_u = [(self.u_t,2.0*v_tp1-1.0)]
        
#        self.update_x = theano.function(inputs=[],
#                                        outputs = [],
#                                     updates = updates_x,
#                                     allow_input_downcast = True,
#                                     on_unused_input='ignore')
#        self.update_y = theano.function(inputs=[],
#                                        outputs = [],
#                                     updates = updates_y,
#                                     allow_input_downcast = True,
#                                     on_unused_input='ignore')
#        self.update_weights = theano.function(inputs = [v_tp1,learning_rate],
#                                     outputs = [],
#                                     updates = updates_gradients,
#                                     allow_input_downcast = True,
#                                     on_unused_input='ignore')
#        self.update_gradients = theano.function(inputs = [v_tp1,learning_rate],
#                                     outputs = [],
#                                     updates = updates_train_weights,
#                                     allow_input_downcast = True,
#                                     on_unused_input='ignore')
#        
        self.update_all_at_once = theano.function(inputs = [v_tp1,learning_rate,
                                                            alpha_param,beta_param],
                                     outputs = [],
                                     updates=updates_x+updates_y+updates_u+updates_train_weights+updates_gradients,
                                     allow_input_downcast = True,
                                     on_unused_input='ignore')
        
    def MakeStep(self,v,learning_rate,alpha=None,beta=None):
        if alpha is None:
            alpha = self.alpha_default_value
        if beta is None:
            beta = self.beta_default_value
        self.update_all_at_once(v,learning_rate,alpha,beta)
    def GetWeightGradients(self):
        result = {}
        gradients = self.get_y_gradients()
        result['dW_xx'] = gradients[0][0,:]
        result['dW_xu'] = gradients[1][0,:]
        result['dW_yx'] = gradients[2][0,:]
        result['db_x'] = gradients[3][0,:]
        result['db_y'] = gradients[4][0,:]
        return result
    def GetCurrentX(self):
        return self.x_t.get_value()
    
    def GetNextX(self):
        return self.get_x_tp1()
    
    def Get_Y_Prediction(self):
        return self.get_y_tp1()
        
    def GetCurrentU(self):
        return self.u_t.get_value()
        
    def Reset(self,nx,randomize=True,sigmas=None):
        if ( sigmas is None ):
            sigmas = self.sigma_default
        self.nx = nx
        if randomize:
            self.W_xu_0 = np.asarray(
                self.rng.normal(size=(self.nu, self.nx), scale= sigmas, loc = .0), dtype = theano.config.floatX)
            self.W_xx_0 = np.asarray(
                self.rng.normal(size=(self.nx, self.nx), scale=sigmas, loc = .0), dtype = theano.config.floatX)
            self.W_yx_0 = np.asarray(
                self.rng.normal(size=(self.nx, self.ny), scale =sigmas, loc=0.0), dtype = theano.config.floatX)
            self.b_x_0 = np.asarray(
                self.rng.normal(size=(self.nx,), scale =sigmas, loc=0.0), dtype = theano.config.floatX)
            self.b_y_0 = np.asarray(
                self.rng.normal(size=(self.ny,), scale =sigmas, loc=0.0), dtype = theano.config.floatX)
            self.x_0 = 1.0*np.random.rand(nx)-0.5
        self.W_xu.set_value(self.W_xu_0)
        self.W_xx.set_value(self.W_xx_0)
        self.W_yx.set_value(self.W_yx_0)
        self.b_x.set_value(self.b_x_0)
        self.b_y.set_value(self.b_y_0)
        
        dx_t_dW_xu_0 = np.zeros(shape=(self.nx,) + self.W_xu_0.shape)
        dx_t_dW_xx_0 = np.zeros(shape=(self.nx,) + self.W_xx_0.shape)
        dx_t_dbx_0 = np.zeros(shape=(self.nx,self.nx))
        
        self.dx_t_dW_xu.set_value(dx_t_dW_xu_0)
        self.dx_t_dW_xx.set_value(dx_t_dW_xx_0)
        self.dx_t_dbx.set_value(dx_t_dbx_0)
        
        self.x_t.set_value(self.x_0)
#        self.y_t.set_value(np.array([1.0,0.0]))
        self.u_t.set_value(np.array([-1.0]))
    def GetState(self):
        D = dict([(W.name,W.get_value()) for W in self.shared_variables])
        D['nx'] = self.nx
        return D
    def SetState(self,state):
        self.nx = state['nx']
        for W in self.shared_variables:
            W.set_value(state[W.name])
        


if __name__ == '__main__':
    
    input_str = '01011110110101001110101000010100010100000010010101010111000110010111101111101010111010101010111101010101010111101011010101010011101111111100101010111111010111010100010110100001111101001110101110101011110101010000011000111000111010101111110101010111110101101011111101101010101010111101011010101010101010111010111110101011110101010011110111110111101010101010000011101011110101000101010000101001101001'
#    input_str = '00010110111101010100111010010100001001110100100010100000101101110101101010110101110111010001010101010101000010100010101001001001011011110101001001001110100100011001010001011100101010101111010010010000101000100100101101000101000000010111001010010100100000101100101110101101010101100010100110101010110101010100101101110101001110101011111110100101001001010011100101010010001000101111001011101001010001010111010101000101011000101010111100101010000100101010011101010111100101010001010100101110100101011111000010101001010010101010101010001010111011010101011011111100110010111100100101010010100011001010110010100010110101111011001010010011101010101101010110101010100101010111001010111010100110001011101010101010111111100101000100101001000000101001001011010100010101010101000010101101001100101001100101111110010101001010'
    
    #input_str = '001001011'*50
    def make_arr(s):
        return np.asarray([0 if c=='0' else 1 for c in s])
    
    #ResetWeights() #useless because gradients are reset
    
    nx = 4
    lr=0.2
    rtrl_nn = RTRL2()
    rtrl_nn.BuildNetwork()
    rtrl_nn.Reset(nx)
#    import cPickle
#    from theano.misc import pkl_utils
#    with open('rtrl_1_1.pickle','w') as f:
#        pk = pkl_utils.Pickler(f)
#        pk.dump(rtrl_nn)
    input_length = len(input_str)
    input_array = make_arr(input_str)
    input_array = input_array.reshape((len(input_array),1))
#    v = np.zeros((input_array.shape[0],2))
#    v[input_array[:,0]==0.0,0]=1.0
#    v[input_array[:,0]==1.0,1]=1.0
    v = input_array
    u_orig=np.array(input_array,copy=True)
    ys = np.nan*np.zeros(v.shape)
    xs = np.nan*np.zeros((v.shape[0],nx))
    for i in xrange(v.shape[0]):
        
#        next_x = rtrl_nn.get_x_tp1(u[i-1,:],current_x)
#        rtrl_nn.update_weights(current_x,u[i-1,:],v[i,:],lr)
#        rtrl_nn.update_gradients(current_x,u[i-1,:],v[i,:],lr)
#        rtrl_nn.update_all_at_once(current_x,u[i-1,:],v[i,:],lr)
        xs[i,:] = rtrl_nn.GetCurrentX()
        ys[i,:] = rtrl_nn.Get_Y_Prediction()
        rtrl_nn.MakeStep(v[i,:],lr)
        
    guessed_results = np.asarray(np.round(ys),dtype=np.int)
    last_half = guessed_results[(input_length/2):]
    successful = np.sum(last_half == u_orig[(input_length/2):])
    print '%d/%d = %.1f%%'%(successful,len(last_half),100*float(successful)/len(last_half))
    print 'X = ',rtrl_nn.GetCurrentX()
#    print 'Cost = ',rtrl_nn.cost_x_deviation(rtrl_nn.GetCurrentX()).eval()
#    x=T.vector('x')
#    c = log_deviation_cost(x).mean()
#    g = T.grad(c,x)
#    func1 = theano.function([x],g)
#    print func1(rtrl_nn.GetCurrentX())