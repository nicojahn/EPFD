# coding: utf-8
import tensorflow as tf

#OMP_NUM_THREADS=1
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

map_kwargs = {"fn_output_signature" : tf.float32, "parallel_iterations" : 30, "swap_memory" : True}
jit_compile = True

# ----------  No changes below this line  -----------

def allow_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)    
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

allow_growth()

minus_one = tf.constant(-1, dtype=tf.int32)
zero = tf.constant(0, dtype=tf.float32)
one = tf.constant(1, dtype=tf.float32)
two = tf.constant(2, dtype=tf.float32)
log_two = tf.math.log(two)
numeric_additive = tf.constant(1e-16, dtype=tf.float32)

# ----------  Probability of Discrete Variable  -----------

# probability of one vector
#
@tf.function(jit_compile=jit_compile)
def prob(X):
    x_ = tf.cast(tf.reduce_sum(X, axis=0), dtype=tf.float32)
    return x_/(tf.reduce_sum(x_) + numeric_additive)

# joint probability of two vectors
#
@tf.function(jit_compile=jit_compile)
def jointProb(X, Y):
    pxy = tf.cast(tf.matmul(tf.transpose(X), Y), dtype=tf.float32)
    return pxy / (tf.reduce_sum(pxy) + numeric_additive)

# ----------  Shannon Entropy  -----------
# calculate values of entropy
# H(.) is the entropy function and p(.,.) is the joint probability

# returns always a semi-positive matrix
@tf.function(jit_compile=jit_compile)
def H(p):
    return -p * tf.math.log(p + numeric_additive) / log_two

# H(X), H(Y) :  for one vector
#
@tf.function(jit_compile=jit_compile)
def H1(X):
    px = prob(X)
    return tf.reduce_sum(H(px))

# H(X,Y) :  for two vectors
#
@tf.function(jit_compile=jit_compile)
def H2e(pxy):
    return tf.reduce_sum(H(pxy))

# Old API call
@tf.function(jit_compile=jit_compile)
def H2(X ,Y):
    pxy = jointProb(X, Y)
    return H2e(pxy)

# I(.;.) is the mutual information function
# I(X; Y)
#
@tf.function(jit_compile=jit_compile)
def Ie(X, Y, pxy):
    px = prob(X)
    py = prob(Y)
    
    num_classes = px.shape[0]

    px = tf.repeat(tf.expand_dims(px, 1), [num_classes], axis=1)
    py = tf.repeat(tf.expand_dims(py, 0), [num_classes], axis=0)

    pxyt = tf.math.divide(pxy, (px * py) + numeric_additive)
    
    return tf.reduce_sum(pxy * tf.math.log(pxyt + numeric_additive) / log_two)

# Old API call
@tf.function(jit_compile=jit_compile)
def I(X, Y):
    pxy = jointProb(X, Y)
    return Ie(X, Y, pxy)

# MI(X, Y): The normalized mutual information of two discrete random variables X and Y
#
@tf.function(jit_compile=jit_compile)
def MI(X, Y):
    pxy = jointProb(X, Y)
    return Ie(X, Y, pxy) / (tf.sqrt(H1(X) * H1(Y)) + numeric_additive)

# VI(X, Y): the normalized variation of information of two discrete random variables X and Y
#
@tf.function(jit_compile=jit_compile)
def VI(X, Y):
    pxy = jointProb(X, Y)
    return one - Ie(X, Y, pxy) / (H2e(pxy) + numeric_additive)

# For two feature vectors like p and q, and the class label vector L, define TDAC(p,q) as follows:
#
@tf.function(jit_compile=jit_compile)
def TDAC(X, Y, L, lam):
    if tf.reduce_all(tf.equal(X, Y)):
        return zero
    return lam * VI(X, Y) + (one - lam) * (MI(X, L) + MI(Y, L)) / two

# S \subset or \subseteq N,  N is the set of all individuals and |S|=k.
# We want to maximize the following objective function (as the objective of diversity maximization problem)
# for `S' \subset `N' and |S|=k
@tf.function(jit_compile=jit_compile)
def TDAS(S, L, lam):
    r = tf.stop_gradient(tf.map_fn(lambda c: tf.map_fn(lambda e: TDAC(c, e, L, lam), S, **map_kwargs), S, **map_kwargs))
    return tf.reduce_sum(r) / two

#----------  Algorithm COMEP  -----------
@tf.function(jit_compile=jit_compile)
def tdac_sum(c, E, L, lam):
    r = tf.stop_gradient(tf.map_fn(lambda e: TDAC(c, e, L, lam), E, **map_kwargs))
    return tf.reduce_sum(r)

# T is the set of individuals; S = [True,False] represents this one is in S or not, and S is the selected individuals.
#
@tf.function
def arg_max_p(T, S_, L, lam):
    
    all_q_in_S = tf.boolean_mask(T, S_, axis=0)
    idx_p_not_S = tf.reshape(tf.cast(tf.where(tf.logical_not(S_)), tf.int32), [-1])
    
    if idx_p_not_S.shape[0] == 0:
        return -1
    
    #r = tf.stop_gradient(tf.map_fn(lambda t: tdac_sum(t, all_q_in_S, L, lam), tf.gather(T, idx_p_not_S), **map_kwargs))
    r = tf.vectorized_map(lambda t: tdac_sum(t, all_q_in_S, L, lam), tf.gather(T, idx_p_not_S))
    
    idx_p = tf.argmax(r, output_type=tf.int32)
    idx = tf.gather(idx_p_not_S, idx_p)
    return idx

# T:    set of individuals
# k:    number of selected individuals
#
@tf.function
def COMEP_(T, k, L, lam, n):
    if n <= k:
        return tf.ones(n, dtype=tf.bool)
    
    idx = tf.random.uniform([1], minval=0, maxval=n, dtype=tf.int32)[0]
    ones = tf.ones(n, dtype=tf.bool)
    S = tf.zeros(n, dtype=tf.bool)
    
    S = tf.where(tf.cast(tf.one_hot(idx, n), tf.bool), ones, S)
    for i in tf.range(1, k):
        idx = arg_max_p(T, S, L, lam)
        if idx > minus_one:
            S = tf.where(tf.cast(tf.one_hot(idx, n), tf.bool), ones, S)
    return S

# ---------- Algorithm DOMEP ----------

# Partition $\mathcal{H}$ (with $n$ individuals inside) randomly
# into $m$ groups as !NOT GUARANTEED! equally as possible.
#
@tf.function(jit_compile=jit_compile)
def randomly_partition(num_samples, num_machines):
    # the idea:
    #partitions = tf.sort(tf.random.categorical(tf.ones((1,num_machines)), num_samples))
    # more 'logically' stable:
    repetitions = tf.cast(tf.math.ceil(num_samples/num_machines), dtype=tf.int32)
    partitions = tf.random.shuffle(tf.repeat(tf.range(num_machines), repetitions))[:num_samples]
    return partitions

@tf.function
def find_idx_in_sub(T, partitions, k, L, lam, n):
    sub_idx_single = COMEP_(T, k, L, lam, n)
    sub_idx_single = tf.reshape(tf.cast(tf.where(sub_idx_single), tf.int32), [-1])
    return tf.gather(partitions, sub_idx_single)

@tf.function
def DOMEP_(T, P, shapes, N, k, m, L, lam):
    # BUG: With tf.stop_gradient results in "ValueError: TypeError: object of type 'RaggedTensor' has no len()"
    r = tf.map_fn(lambda i: find_idx_in_sub(tf.gather(T, i), tf.gather(P, i), k, L, lam, tf.gather(shapes, i)), tf.range(m), fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32), parallel_iterations=m, swap_memory=True)

    idx = r.values
    T_sup = tf.gather(N, idx)
    
    if m > 1:
        sub_n = tf.reduce_sum(tf.minimum(k, shapes))
        sub_all_single = COMEP_(T_sup, k, L, lam, sub_n)
        sub_all_single = tf.reshape(tf.cast(tf.where(sub_all_single), tf.int32), [-1])
        idx_sup = tf.gather(idx, sub_all_single)

        all_sets = tf.concat([tf.expand_dims(idx_sup, axis=0), r], axis=0)
        tdas_final = tf.stop_gradient(tf.map_fn(lambda i: TDAS(tf.gather(N, i), L, lam), all_sets, **map_kwargs))

        idx_p = tf.argmax(tdas_final, output_type=tf.int32)
        solution = tf.gather(all_sets, idx_p)
    else:
        solution = r[0]

    n = tf.reduce_sum(shapes)
    final_S = tf.cast(tf.reduce_sum(tf.one_hot(solution, n), axis=0), dtype=tf.bool)
    return final_S

# ---------- Preparation Wrapper ----------


def ToOneHot(T, L, num_classes):
    return (
        tf.cast(tf.transpose(tf.one_hot(T, num_classes, axis=1), (0,2,1)), dtype=tf.int32), 
        tf.cast(tf.transpose(tf.one_hot(L, num_classes, axis=0), (1,0)), dtype=tf.int32)
    )

def COMEP(T, k, L, lam, num_classes=2):
    T_ = tf.constant(T)
    L_ = tf.constant(L)
    if T_.shape[0] == L_.shape[0]:
        T_ = tf.transpose(T_, (1, 0))
    T_, L_ = ToOneHot(T_, L_, num_classes)
    
    nb_cls = T_.shape[0]
    nb_pru = k
    
    return COMEP_(T_, nb_pru, L_, lam, nb_cls)

# tf.dynamic_partition is not completely 'dynamic', therefore no tf.function
def DOMEP(T, k, m, L, lam, num_classes=2):
    T_ = tf.constant(T)
    L_ = tf.constant(L)
    if T_.shape[0] == L_.shape[0]:
        T_ = tf.transpose(T_, (1, 0))
    T_, L_ = ToOneHot(T_, L_, num_classes)

    nb_cls = T_.shape[0]
    nb_pru = k

    partitions = randomly_partition(nb_cls, m)
    Tp_ = tf.ragged.stack_dynamic_partitions(T_, partitions, m)

    p = tf.range(nb_cls)
    P_ = tf.ragged.stack_dynamic_partitions(p, partitions, m)
    shapes = tf.stop_gradient(tf.map_fn(lambda x: x.shape[0], P_, fn_output_signature=tf.int32))

    return DOMEP_(Tp_, P_, shapes, T_, nb_pru, m, L_, lam)