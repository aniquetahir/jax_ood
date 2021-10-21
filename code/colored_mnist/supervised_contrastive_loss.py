import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, jit
from jax.experimental import loops


tau = 1.

def scl_numerator(zi, zp):
    return jnp.exp(jnp.dot(zi, zp)/tau)


def scl_denominator(zi, za, mask):
    vec_scl_exp = jax.vmap(scl_numerator, in_axes=(None, 0))
    denom = vec_scl_exp(zi, za) * mask
    denom = jnp.sum(denom)
    return denom


def scl_inner(zi, zp, zp_mask, embeddings, except_mask):
    numerator = scl_numerator(zi, zp) * zp_mask
    # flattened_embeddings = embeddings
    # num_embeddings = flattened_embeddings.shape[0]
    # zi_matrix = zi * np.ones((num_embeddings, 1))
    denominator = scl_denominator(zi, embeddings, except_mask)
    return jnp.sum(jnp.log(numerator/denominator))


def sup_contrastive_loss_i(zi_index, zi_label, embeddings, labels):
    # num_envs = envs.shape[0]
    # get number of positive indices(excluding zi)
    #   get mask for same class
    # with loops.Scope() as s:
    #     s.env_masks = jnp.zeros(envs.shape[:2])
    #     for i_env in s.range(s.env_masks.shape[0]):
    #         np.where(envs[i_env]['labels'] == zi_label, 1, 0)
    #     # s.env_masks = s.env_masks.at[zi_env, zi_index].set(0.)
    label_masks = jnp.where(labels == zi_label, 1, 0)
    label_masks = label_masks.at[zi_index].set(0.)
    except_mask = jnp.ones_like(label_masks)
    except_mask = except_mask.at[zi_index].set(0)
    sim = label_masks
    diff = jnp.ones_like(sim) - sim
    num_sim = jnp.sum(sim)

    # get zi, zp
    zi = embeddings[zi_index]
    z_all = embeddings
    zp_mask = label_masks
    vec_scl_inner = jax.vmap(scl_inner, in_axes=(None, 0, 0, None, None))
    inner_loss = vec_scl_inner(zi, z_all, zp_mask, embeddings, except_mask)
    return (-1./num_sim) * inner_loss


def get_sup_contrastive_loss_fn(mlp_fts):
    return partial(sup_contrastive_loss, mlp_fts=mlp_fts)

def sup_contrastive_loss(params, env_images, env_labels, mlp_fts):
    # get_embeddings
    embeddings = jnp.vstack(env_images)
    embeddings = mlp_fts(params, embeddings)
    labels = jnp.vstack(env_labels)

    # vectorize z_i
    zi_indices = jnp.arange(labels.shape[0])
    loss_outer = jax.vmap(sup_contrastive_loss_i, in_axes=(0, 0, None, None))

    # sum over loss
    loss = loss_outer(zi_indices, labels, embeddings, labels)