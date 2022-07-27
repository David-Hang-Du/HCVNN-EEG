import tensorflow as tf
from scipy.stats import rayleigh, uniform
import  numpy as np
import sys
from time import sleep
from sklearn.metrics import confusion_matrix

def convolution(X, W, b, padding, stride):
    #n, h, w, c = map(lambda d: d.value, X.get_shape())
    #filter_h, filter_w, filter_c, filter_n = [d.value for d in W.get_shape()]
    n, h, w, c = X.shape
    filter_h, filter_w, filter_c, filter_n = W.get_shape()
    
    out_h = (h + 2*padding - filter_h)//stride + 1
    out_w = (w + 2*padding - filter_w)//stride + 1

    X_flat = flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
    W_flat = tf.reshape(W, [filter_h*filter_w*filter_c, filter_n])
    
    #print(X_flat)
    #print(W_flat)
    
    z = tf.matmul(X_flat, W_flat) + b     # b: 1 X filter_n
    
    return tf.transpose(tf.reshape(z, [out_h, out_w, n, filter_n]), [2, 0, 1, 3])

def flatten(X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    
    X_padded = tf.pad(X, [[0,0], [padding, padding], [padding, padding], [0,0]])

    windows = []
    for y in range(out_h):
        for x in range(out_w):
            window = tf.slice(X_padded, [0, y*stride, x*stride, 0], [-1, window_h, window_w, -1])
            windows.append(window)
    stacked = tf.stack(windows) # shape : [out_h, out_w, n, filter_h, filter_w, c]

    return tf.reshape(stacked, [-1, window_c*window_w*window_h])

def relu_real(X):
    return tf.maximum(X, tf.zeros_like(X))

def cartReLu(X):
    real = tf.math.real(X)
    imag = tf.math.imag(X)
    return tf.complex(tf.maximum(real, tf.zeros_like(real)), tf.maximum(imag, tf.zeros_like(imag)))
    
def max_pool_real(X, pool_h, pool_w, padding, stride):
    n, h, w, c = X.get_shape()
    
    out_h = (h + 2*padding - pool_h)//stride + 1
    out_w = (w + 2*padding - pool_w)//stride + 1

    X_flat = flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)

    pool = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h*pool_w, c]), axis=3)
    return tf.transpose(pool, [2, 0, 1, 3])


def max_pool_real_new(X, pool_h, pool_w, padding, stride):
    output, argmax = tf.nn.max_pool_with_argmax(input=X, ksize=(pool_h,pool_w), strides=stride,
                                                padding='VALID', include_batch_in_index=True)
    # shape = tf.shape(output)
    # tf_res = tf.reshape(tf.gather(tf.reshape(X, [-1]), argmax), shape)
    # assert np.all(tf_res == output)             # For debugging when the input is real only!
    # assert tf_res.dtype == inputs.dtype
    return output


def max_pool_complex(X, pool_h, pool_w, padding, stride):
    n, h, w, c = X.get_shape()
    
    out_h = (h + 2*padding - pool_h)//stride + 1
    out_w = (w + 2*padding - pool_w)//stride + 1

    X_flat = flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)
    X_reshape = tf.reshape(X_flat, [out_h, out_w, n, pool_h*pool_w, c])
    X_reshape_abs = tf.abs(X_reshape)
    X_reshape_abs_max = tf.where(tf.equal(tf.reduce_max(X_reshape_abs, axis=3, keepdims=True), X_reshape_abs),
                    tf.constant(1, shape=X_reshape_abs.shape),tf.constant(0, shape=X_reshape_abs.shape))
    X_reshape_max = tf.dtypes.cast(X_reshape_abs_max, tf.complex64)*X_reshape
    X_complex = tf.complex(tf.reduce_max(tf.math.real(X_reshape_max), axis=3)+tf.reduce_min(tf.math.real(X_reshape_max), axis=3), 
                           tf.reduce_max(tf.math.imag(X_reshape_max), axis=3)+tf.reduce_min(tf.math.imag(X_reshape_max), axis=3))

    return tf.transpose(X_complex, [2, 0, 1, 3])

def dense(X, W, b):
    n = X.get_shape()[0] # number of samples
    X_flat = tf.reshape(X, [n, -1])
    #print(X_flat.get_shape())
    return tf.matmul(X_flat, W) + b

def dropout_complex(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = tf.dtypes.cast(tf.random.uniform(X.shape) < keep_probability, dtype=tf.complex64)
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * tf.dtypes.cast(scale, dtype=tf.complex64)

def dropout_real(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = np.random.uniform(0.,1.,X.shape) < keep_probability
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale

# complex kernel initialization
class layer_init_complex:
    def __init__(self, size, std):
        self.size = size
        self.sd = std
        #np.random.seed(seed=std)
        if len(self.size)<4:
            r = rayleigh.rvs(size=self.size, scale=(1/np.sqrt(self.size[0]*self.size[1])))
        else:
            r = rayleigh.rvs(size=self.size, scale=(1/np.sqrt(self.size[0]*self.size[1]*self.size[3])))
            
        u = uniform.rvs(size=self.size, loc = -np.pi, scale = 2*np.pi)
        self.ker = tf.Variable(tf.dtypes.cast(r*np.exp(u*1j), dtype=tf.complex64))
        self.bias = tf.Variable(tf.dtypes.cast(tf.complex(tf.zeros([self.size[-1]]), 
                                                         tf.zeros([self.size[-1]])), 
                                                        dtype=tf.complex64))
        self.ker_m = tf.complex(tf.zeros(self.size), tf.zeros(self.size))
        self.ker_v = tf.complex(tf.zeros(self.size), tf.zeros(self.size))
        self.bias_m = tf.complex(tf.zeros([self.size[-1]]), tf.zeros([self.size[-1]]))
        self.bias_v = tf.complex(tf.zeros([self.size[-1]]), tf.zeros([self.size[-1]]))
        self.n = 0.
    def update(self, grad_ker, grad_bias, learning_rate, beta1, beta2, epsilon):
      ########################
      ###### check here ######
      ########################
        self.n = self.n + 1.
        self.ker_m = beta1*self.ker_m + (1-beta1)*grad_ker#.numpy()
        self.ker_v = beta2*self.ker_v + tf.complex((1-beta2)*tf.math.abs(grad_ker)**2,tf.zeros(self.size, dtype=tf.dtypes.float32))
        self.ker.assign(self.ker - learning_rate * self.ker_m / (1-beta1**self.n)/((self.ker_v/(1-beta2**self.n))**0.5+epsilon))
        
        if grad_bias is not None:
            self.bias_m = beta1*self.bias_m + (1-beta1)*grad_bias#.numpy()
            self.bias_v = beta2*self.bias_v + tf.complex((1-beta2)*tf.math.abs(grad_bias)**2,tf.zeros([self.size[-1]], dtype=tf.dtypes.float32))
            self.bias.assign(self.bias - learning_rate * self.bias_m / (1-beta1**self.n)/((self.bias_v/(1-beta2**self.n))**0.5+epsilon))

# real kernel initialization        
class layer_init_real:
    def __init__(self, size, sd):
        self.size = size
        # self.std = std
        if len(self.size)<4:
            std = np.sqrt(6/(self.size[0]+self.size[1]))
        else:
            std = np.sqrt(6/(self.size[0]*self.size[1]*(self.size[2] + self.size[3])))
        # self.ker = tf.Variable(tf.dtypes.cast(tf.random.normal(self.size, stddev = self.std), dtype=tf.float32))
        self.ker = tf.Variable(tf.dtypes.cast(uniform.rvs(size=self.size, loc = -std, scale = 2*std), dtype=tf.float32))
        self.bias = tf.Variable(tf.dtypes.cast(tf.zeros([self.size[-1]]), dtype=tf.float32))
        self.ker_m = tf.zeros(self.size)
        self.ker_v = tf.zeros(self.size)
        self.bias_m = tf.zeros([self.size[-1]])
        self.bias_v = tf.zeros([self.size[-1]])
        self.n = 0.
    def update(self, grad_ker, grad_bias, learning_rate, beta1, beta2, epsilon):
      ########################
      ###### check here ######
      ########################
        self.n = self.n + 1.
        self.ker_m = beta1*self.ker_m + (1-beta1)*grad_ker
        self.ker_v = beta2*self.ker_v + (1-beta2)*grad_ker**2
        self.ker.assign(self.ker - learning_rate * self.ker_m / (1-beta1**self.n)/((self.ker_v/(1-beta2**self.n))**0.5+epsilon))
        if grad_bias is not None:
            self.bias_m = beta1*self.bias_m + (1-beta1)*grad_bias
            self.bias_v = beta2*self.bias_v + (1-beta2)*grad_bias**2
            self.bias.assign(self.bias - learning_rate * self.bias_m / (1-beta1**self.n)/((self.bias_v/(1-beta2**self.n))**0.5+epsilon))
            
            
def train(layer_comb,x_train, y_train, train_vars, vars_name, epoch = 50, train_batch_size=32, decay=False, decay_rate=0.99):
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    loop_size = int(x_train.shape[0]/train_batch_size)
    loss_epoch = np.zeros([epoch])
    loss_total = np.zeros([loop_size*epoch])
    acc_epoch = np.zeros([epoch])
    

    for j in range(epoch):
        
        if decay is True:
            learning_rate = decay_rate**j*learning_rate
        
        
        k = (j + 1) / epoch
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('='*int(50*k), 100*k))
        sys.stdout.flush()
        sleep(0.)
        diff_XY_total = 0.
        loss_tem = 0.
        # shaffling
        index = np.random.permutation(x_train.shape[0])
        x_train = np.take(x_train, index, axis=0)
        y_train = np.take(y_train, index, axis=0)

        for i in range(loop_size):
            
            x_train_tem = x_train[train_batch_size*i:train_batch_size*(i+1),:,:,:]
            y_train_tem = y_train[train_batch_size*i:train_batch_size*(i+1),:]
            y_label_tem = tf.cast(tf.argmax(y_train_tem, axis=1), dtype=tf.float32)

            train_results = layer_comb(x_train_tem, y_train_tem, train_vars, vars_name)

            # Weights update

            for kk in range(len(train_vars)):
                ker_name = 'ker_{}'.format(kk)
                bias_name = 'bias_{}'.format(kk)
                train_vars[kk].update(train_results['grad'][ker_name], train_results['grad'][bias_name], learning_rate, beta1, beta2, epsilon)

            diff_XY_total += tf.math.count_nonzero(y_label_tem-train_results['X_label']).numpy()
            loss_tem += train_results['loss'].numpy()
            loss_total[j*loop_size+i] = train_results['loss'].numpy()

        if x_train.shape[0]%train_batch_size is not 0:
            x_train_tem = x_train[train_batch_size*loop_size:,:,:,:]
            y_train_tem = y_train[train_batch_size*loop_size:,:]
            y_label_tem = tf.cast(tf.argmax(y_train_tem, axis=1), dtype=tf.float32)

            train_results = layer_comb(x_train_tem, y_train_tem, train_vars, vars_name)

            # Weights update

            for kk in range(len(train_vars)):
                ker_name = 'ker_{}'.format(kk)
                bias_name = 'bias_{}'.format(kk)
                train_vars[kk].update(train_results['grad'][ker_name], train_results['grad'][bias_name], learning_rate, beta1, beta2, epsilon)

            diff_XY_total += tf.math.count_nonzero(y_label_tem-train_results['X_label']).numpy()
            loss_tem += train_results['loss'].numpy()
            
        acc_epoch[j] = 1 - diff_XY_total/y_train.shape[0]
        # This average loss may be calculated in a wrong way
        loss_epoch[j] =  loss_tem/loop_size

    return {'acc_epoch':acc_epoch, 'loss_epoch':loss_epoch, 'loss_total':loss_total}

def test(layer_comb, x_test, y_test, train_vars, vars_name, test_batch_size = 100):
  
    diff_XY_total = 0.
    num_class = y_test.shape[1]
    conf_mat = conf_mat = np.zeros((num_class, num_class))
    test_loop_size = int(x_test.shape[0]/test_batch_size)
    for i in range(test_loop_size):

        x_test_tem = x_test[test_batch_size*i:test_batch_size*(i+1),:,:,:]
        y_test_tem = y_test[test_batch_size*i:test_batch_size*(i+1),:]
        y_label_tem = tf.cast(tf.argmax(y_test_tem, axis=1), dtype=tf.float32)

        test_result = layer_comb(x_test_tem, y_test_tem, train_vars, vars_name)
        
        diff_XY_total += tf.math.count_nonzero(y_label_tem-test_result['X_label']).numpy()
        conf_mat += confusion_matrix(y_label_tem, test_result['X_label'].numpy(), labels=range(num_class))

    if x_test.shape[0]%test_batch_size is not 0:

        x_test_tem = x_test[test_batch_size*test_loop_size:,:,:,:]
        y_test_tem = y_test[test_batch_size*test_loop_size:,:]
        y_label_tem = tf.cast(tf.argmax(y_test_tem, axis=1), dtype=tf.float32)

        test_result = layer_comb(x_test_tem, y_test_tem, train_vars, vars_name)

        diff_XY_total += tf.math.count_nonzero(y_label_tem-test_result['X_label']).numpy()
        conf_mat += confusion_matrix(y_label_tem, test_result['X_label'].numpy(), labels=range(num_class))

    acc = 1 - diff_XY_total/y_test.shape[0]
    return {'acc':acc, 'conf_mat':conf_mat}
