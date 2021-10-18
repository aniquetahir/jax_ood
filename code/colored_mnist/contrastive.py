import random

import haiku as hk
from jax.nn import relu
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from optax import adam, sigmoid_binary_cross_entropy
import optax
from torchvision import datasets
f32 = jnp.float32
f64 = jnp.float64
i64 = jnp.int64


hidden_dim =   390 #@param {type:"integer"}
l2_regularizer_weight = 0.00110794568 #@param {type:"number"}
lr = 0.0004898536566546834 #@param {type:"number"}
n_restarts = 10 #@param {type:"integer"}
penalty_anneal_iters = 190 #@param {type:"integer"}
penalty_weight = 91257.18613115903 #@param {type:"number"}
steps = 501 #@param {type:"integer"}
grayscale_model = False #@param ["True", "False"] {type:"raw"

contrastive_tau = 1.0
triplet_loss_weight = 1.
num_triplets = 1000



def cosine_similarity(a, b):
    sim = (jnp.dot(a, b))/(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return sim

@vmap
def similarity_fn(a, b):
    return cosine_similarity(a, b)


class MLP_FTS(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        w_init1 = hk.initializers.RandomUniform(minval=-1)
        b_init1 = hk.initializers.Constant(0.)
        self.seq = hk.Sequential([
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
        ])

    def __call__(self, x):
        inp = x.reshape(x.shape[0], 2*14*14)
        return self.seq(inp)



class MLP(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # self.output_size = output_size
        w_init1 = hk.initializers.RandomUniform(minval=-1)
        b_init1 = hk.initializers.Constant(0.)
        self.seq = hk.Sequential([
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(1)
            ])

    def __call__(self, x):
        inp = x.reshape(x.shape[0], 2*14*14)
        return self.seq(inp)

def mlp_params_to_fts(params):
    tmp = hk.data_structures.to_mutable_dict(params)
    new_params = {}
    for i, v in tmp.items():
        new_params[i.replace('mlp', 'mlp_fts')] = v
    return hk.data_structures.to_immutable_dict(new_params)


# @jit
def mlp_fn(x):
    mlp = MLP(name='simplenn')
    return mlp(x)


def mlp_fts_fn(x):
    mlp = MLP_FTS(name='simplenn')
    return mlp(x)


init, apply = hk.without_apply_rng(hk.transform(mlp_fn))
_, apply_fts = hk.without_apply_rng(hk.transform(mlp_fts_fn))

key = jax.random.PRNGKey(6)
# key, split = jax.random.split(key)


@jit
def mean_nll(logits, y):
    return jnp.mean(sigmoid_binary_cross_entropy(logits, y))


@jit
def mean_accuracy(logits, y):
    preds = (logits > 0.).astype(float)
    return jnp.mean((jnp.abs((preds - y)) < 1e-2).astype(float))


@jit
def penalty(logits, y):
    nll_grad = jax.grad(mean_nll)
    grad = nll_grad(logits, y)
    return jnp.sum(grad**2)

@jit
def loss_fn(params, envs):
    def aggregator(c, x):
        env_images = jnp.array(c['env_images'])[x]
        env_labels = jnp.array(c['env_labels'])[x]
        # actual_samples = c['env_scale'][x]
        # env_images = env_images[:actual_samples]
        # env_labels = env_labels[:actual_samples]
        logits = apply(params, env_images)
        carry = {
            'nll': c['nll'].at[x].set(mean_nll(logits, env_labels)),
            'acc': c['acc'].at[x].set(mean_accuracy(logits, env_labels)),
            'penalty': c['penalty'].at[x].set(penalty(logits, env_labels)),
            'env_images': c['env_images'],
            'env_labels': c['env_labels']
        }

        return carry, x

    losses, _ = jax.lax.scan(aggregator, {
        'nll': jnp.zeros(len(envs)),
        'acc': jnp.zeros(len(envs)),
        'penalty': jnp.zeros(len(envs)),
        'env_images': [x['images'] for x in envs],
        'env_labels': [x['labels'] for x in envs]
    }, jnp.arange(len(envs)))

    train_nll = jnp.mean(losses['nll'])
    train_acc = jnp.mean(losses['acc'])
    train_penalty = jnp.mean(losses['penalty'])

    weight_norm = jnp.sum(jnp.stack([jnp.linalg.norm(x['w'])**2 for x in params.values()]))
    loss = train_nll
    loss += l2_regularizer_weight * weight_norm

    penalty_weight = (
        penalty_weight if step >= penalty_anneal_iters else 1.0
    )

    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
        loss /= penalty_weight

    return (loss, (losses, train_nll, train_acc, train_penalty))


def triplet_loss(params, triplet_images, triplet_labels):
    # mlp_params_to_fts(params)
    # num_envs = labels.shape[0]
    t1_preds = apply_fts(params, triplet_images[0])
    t2_preds = apply_fts(params, triplet_images[1])
    t3_preds = apply_fts(params, triplet_images[2])

    numerator = jnp.exp(similarity_fn(t1_preds, t2_preds)/contrastive_tau)
    denonminator = jnp.exp(similarity_fn(t1_preds, t3_preds)/contrastive_tau)

    loss = -jnp.log(numerator/denonminator)
    return jnp.mean(loss)


#@jit
def ultimate_loss(params, envs, triplet_images):
    irm_loss, (losses, train_nll, train_acc, train_penalty) = loss_fn(params, envs)
    t_loss = triplet_loss(params, triplet_images, None)
    ult_loss = irm_loss + triplet_loss_weight * t_loss
    return ult_loss, (losses, train_nll, train_acc, train_penalty)


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

from collections import defaultdict

def create_triplets(num_samples, envs, sim_fn):
    num_envs = len(envs)
    num_source = envs[0]['labels'].shape[0]
    choice_triplet_envs = np.random.choice(np.arange(num_envs), 3)
    sample_range = np.arange(num_samples)
    t1 = np.random.choice(np.arange(num_source), num_samples, replace=False)
    choice_labels = envs[choice_triplet_envs[0]]['labels'][t1]

    # Select secondlet
    t2_indices = []
    t2_env = envs[choice_triplet_envs[1]]
    t2_label_mapping = defaultdict(list)
    for i, label in enumerate(t2_env['labels'].tolist()):
        t2_label_mapping[tuple(label)].append(i)

    while len(t2_indices) < num_samples:
        choice_label = tuple(choice_labels[len(t2_indices)])
        sec_index = np.random.choice(t2_label_mapping[choice_label])
        t2_indices.append(sec_index)
        # sec_index = np.random.choice(sample_range)
        # if envs[choice_triplet_envs[1]]['labels'][sec_index] == choice_labels[len(t2_indices)]:
        #     t2_indices.append(sec_index)

    t3_indices = []
    t3_env = envs[choice_triplet_envs[2]]
    t3_label_mapping = defaultdict(list)
    for i, label in enumerate(t3_env['labels'].tolist()):
        t3_label_mapping[tuple(label)].append(i)

    while len(t3_indices) < num_samples:
        choice_label = tuple(choice_labels[len(t3_indices)])
        other_labels = [x for x in t3_label_mapping.keys() if x != choice_label]
        other_label = random.choice(other_labels)
        t3_indices.append(np.random.choice(t3_label_mapping[other_label]))

        # ter_index = np.random.choice(sample_range)
        # if envs[choice_triplet_envs[2]]['labels'][ter_index] != choice_labels[len(t3_indices)]:
        #     t3_indices.append(ter_index)

    # num_same_t2, num_diff_t2
    image_triplets = np.array([envs[choice_triplet_envs[0]]['images'][t1],
                               envs[choice_triplet_envs[1]]['images'][np.array(t2_indices)],
                               envs[choice_triplet_envs[2]]['images'][np.array(t3_indices)]])

    label_triplets = np.array([envs[choice_triplet_envs[0]]['labels'][t1],
                               envs[choice_triplet_envs[1]]['labels'][np.array(t2_indices)],
                               envs[choice_triplet_envs[2]]['labels'][np.array(t3_indices)]])

    return image_triplets, label_triplets, choice_triplet_envs


if __name__ == "__main__":
    # params = init(split, envs)
    # optim = adam(lr)
    # opt_state = optim.init(params)
    for restart in range(n_restarts):
        print("Restart", restart)
        mnist = datasets.MNIST('datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])

        mnist_train = tuple(map(lambda x: np.array(x), mnist_train))
        mnist_val = tuple(map(lambda x: np.array(x), mnist_val))

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0])
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1])

        @jit
        def make_environment(key, images, labels, e):
            def bernoulli(key, p, size):
                key, split = jax.random.split(key)
                return (jax.random.uniform(split, (size,)) < p).astype(float)

            def xor(a, b):
                return jnp.abs(a - b)

            key, split = jax.random.split(key)
            images = images.reshape((-1, 28, 28))[:, ::2, ::2]
            # Ass binary labels based on the digit, flip label
            labels = (labels<5).astype(float)
            labels = xor(labels, bernoulli(split, 0.25, len(labels)))
            # Ass color , flip with prob e
            key, split = jax.random.split(key)
            colors = xor(labels, bernoulli(split, e, len(labels)))
            images = jnp.stack([images, images], axis=1)

            zeros_set = images[jnp.array(range(len(images))), jnp.array((1-colors), dtype=i64), :, :] * 0
            images.at[jnp.array(range(len(images))), jnp.array((1-colors), dtype=i64), :, :].set(zeros_set)
            # images[jnp.array(range(len(images))), jnp.array((1-colors), dtype=i64), :, :]*= 0
            return {
                'images': (images.astype(float)/255.),
                'labels': labels[:, None]
            }

        key, split1 = jax.random.split(key)
        key, split2 = jax.random.split(key)
        key, split3 = jax.random.split(key)
        envs = [
            make_environment(key, mnist_train[0][::2], mnist_train[1][::2], 0.2),
            make_environment(key, mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
            make_environment(key, mnist_val[0], mnist_val[1], 0.9)
        ]

        # triplet_images, triplet_labels, triplet_envs = create_triplets(1000, envs, None)   # envs[0]['labels'].shape[0], envs, None)

        # input('enter key')
        # Initialize the neural network parameters
        key, split = jax.random.split(key)
        params = init(split, envs[0]['images'][:10])
        optim = optax.adam(lr)
        opt_state = optim.init(params)


        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')


        # def loss_fn_for_grad(params, envs):
        #     loss, _, _, _, _ = loss_fn(params, envs[:2], envs[-1])
        #     return loss
        for step in range(steps):
            triplet_images, triplet_labels, triplet_envs = create_triplets(num_triplets, envs, None)   # envs[0]['labels'].shape[0], envs, None)
            # t_params = mlp_params_to_fts(params)
            # t_loss = triplet_loss(params, triplet_images, None)
            # t_loss = p_triplet_loss(params, triplet_images)
            # loss_grad_fn = jax.grad(loss_fn, has_aux=True)
            loss_grad_fn = jax.grad(ultimate_loss, has_aux=True)
            grads, (values, train_nll, train_acc, train_penalty) = loss_grad_fn(params, envs[:-1], triplet_images)

            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % 100 == 0:
                _, (values, _, test_acc, _) = loss_fn(params, envs[-1:])
                # test_acc = values['acc'][0]
                pretty_print(
                    np.int32(step),
                    train_nll,
                    train_acc,
                    train_penalty,
                    test_acc
                )
        pass
