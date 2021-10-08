import haiku as hk
from jax.nn import relu
import jax
from jax import jit
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

class MLP(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # self.output_size = output_size
        w_init1 = hk.initializers.RandomUniform(minval=-1)
        b_init1 = hk.initializers.Constant(0.)
        self.seq = hk.Sequential([
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(hidden_dim, w_init=w_init1, b_init=b_init1), relu,
                hk.Linear(1)
            ])

    def __call__(self, x):
        return self.seq(x)


# @jit
def mlp_fn(x):
    mlp = MLP()
    return mlp(x)


init, apply = hk.without_apply_rng(hk.transform(mlp_fn))
key = jax.random.PRNGKey(6)
# key, split = jax.random.split(key)


@jit
def mean_nll(logits, y):
    return sigmoid_binary_cross_entropy(logits, y)


@jit
def mean_accuracy(logits, y):
    preds = (logits > 0.).astype(float)
    return jnp.mean((jnp.abs((preds - y)) < 1e-2).astype(float))


@jit
def penalty(logits, y):
    loss_grad = jax.grad(mean_nll)
    grad = loss_grad(logits, y)
    return jnp.sum(grad**2)


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


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


        # Scale the envs using padding to make the jax jit functions work
        scaled_envs = []
        scale_mask = []
        # max_samples = envs[0]['images'].shape[0]
        for env in envs:
            num_samples = env['images'].shape[0]
            scale_mask.append(num_samples)
        max_samples = max(scale_mask)
        for i, env in enumerate(envs):
            sample_difference = max_samples - scale_mask[i]
            if sample_difference == 0:
                continue
            pad_img_data = jnp.zeros(jax.tree_flatten([sample_difference, env['images'].shape[1:]])[0])
            pad_label_data = jnp.zeros(jax.tree_flatten([sample_difference, env['labels'].shape[1:]])[0])
            scaled_envs.append({
                'images': jnp.vstack((env['images'], pad_img_data)),
                'labels': jnp.vstack((env['labels'], pad_label_data))
            })

        envs = scaled_envs

        # Initialize the neural network parameters
        key, split = jax.random.split(key)
        params = init(split, envs[0]['images'][:10])
        optim  = optax.adam(lr)
        opt_state = optim.init(params)


        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

        def loss_fn(params, envs):
            def aggregator(c, x):
                env_images = jnp.array(c['env_images'])[x]
                env_labels = jnp.array(c['env_labels'])[x]
                actual_samples = c['env_scale'][x]
                env_images = env_images[:actual_samples]
                env_labels = env_labels[:actual_samples]
                logits = apply(params, env_images)
                carry = {
                    'nll': c['nll'].at[x].set(mean_nll(logits, env_labels)),
                    'acc': c['acc'].at[x].set(mean_accuracy(logits, env_labels)),
                    'penalty': c['penalty'].at[x].set(penalty(logits, env_labels)),
                    'envs': envs
                }

                return carry, x

            losses, _ = jax.lax.scan(aggregator, {
                'nll': jnp.zeros(3),
                'acc': jnp.zeros(3),
                'penalty': jnp.zeros(3),
                'env_images': [x['images'] for x in envs],
                'env_labels': [x['labels'] for x in envs],
                'env_scale': jnp.array(scale_mask)
            }, jnp.arange(3))

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
            if penalty > 1.0:
                loss /= penalty_weight

            return (loss, (losses, train_nll, train_acc, train_penalty))

        def loss_fn_for_grad(params, envs):
            loss, _, _, _, _ = loss_fn(params, envs)
            return loss


        for step in range(steps):
            loss_grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, (values, train_nll, train_acc, train_penalty) = loss_grad_fn(params, envs)


            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)


            if step % 100 == 0:
                # _, values, train_nll, train_acc, train_penalty = loss_fn(params, envs)
                test_acc = values['acc'][2]
                pretty_print(
                    np.int32(step),
                    train_nll,
                    train_acc,
                    train_penalty,
                    test_acc
                )
        pass
