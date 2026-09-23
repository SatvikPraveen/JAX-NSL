Basic usage
===========

PRNG sequences and initialisers
-------------------------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.core import PRNGSequence, he_normal_init, compute_fans

   rng = PRNGSequence(42)
   w = he_normal_init(next(rng), (16, 3, 3, 3), in_axis=1, out_axis=0)   # conv kernel (O, I, kh, kw)
   print(compute_fans(w.shape, in_axis=1, out_axis=0))                    # (27, 144): receptive field counted

Losses and a jitted training step
---------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.models import create_mlp
   from jax_nsl.training import (adamw_optimizer, cross_entropy_loss, create_learning_rate_schedule,
                                 create_train_state, make_train_step, make_eval_step)

   params, forward, _ = create_mlp([8, 32, 4], init_type="he", seed=0)
   schedule = create_learning_rate_schedule("warmup_cosine", base_lr=3e-3, warmup_steps=20, total_steps=200)
   init, update = adamw_optimizer(schedule, weight_decay=0.01)
   state = create_train_state(params, init, jax.random.PRNGKey(0))

   train_step = make_train_step(forward, cross_entropy_loss, update, max_grad_norm=1.0)
   eval_step = make_eval_step(forward, cross_entropy_loss)

   x = jax.random.normal(jax.random.PRNGKey(1), (128, 8))
   y = jnp.argmax(x[:, :4], axis=1)
   batch = {"inputs": x, "labels": y}
   for _ in range(200):
       state, metrics = train_step(state, batch)
   print(eval_step(state.params, batch))   # {'val_loss': ..., 'val_accuracy': ...}

Gradient checking
-----------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.autodiff import gradient_check_report

   f = lambda x: jnp.sum(jnp.exp(x) * jnp.sin(x))
   print(gradient_check_report(f, jnp.array([0.1, 0.5, 1.0])))
   # complex-step error ~1e-7, central ~1e-4, forward ~1e-2 (float32)

Checkpoints
-----------

.. code-block:: python

   import jax
   from jax_nsl.models import create_mlp
   from jax_nsl.training import adam_optimizer, create_train_state, save_checkpoint, load_checkpoint

   params, forward, _ = create_mlp([8, 32, 4], seed=0)
   state = create_train_state(params, adam_optimizer(1e-3)[0], jax.random.key(0))  # typed key is fine
   path = save_checkpoint(state, "/tmp/jax_nsl_ckpt", epoch=1)
   restored = load_checkpoint(path)
   assert int(restored.step) == int(state.step)
