Autodiff API
============

The ``autodiff`` package provides safe gradient utilities and custom forward/backward differentiation rules.

.. contents:: Modules
   :local:
   :depth: 1

autodiff.grad_jac_hess
----------------------

Higher-order derivative utilities.

.. automodule:: autodiff.grad_jac_hess
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``safe_grad(fun, argnums)``
     - ``jax.grad`` with NaN/Inf detection.
   * - ``compute_gradient(fun, x)``
     - Convenience wrapper – returns ``grad(fun)(x)`` directly.
   * - ``compute_jacobian(fun, x)``
     - Convenience wrapper – returns full Jacobian matrix.
   * - ``compute_hessian(fun, x)``
     - Convenience wrapper – returns full Hessian matrix.
   * - ``gradient_checker(fun, x, h)``
     - Compares analytic vs finite-difference gradients.

autodiff.custom_vjp
-------------------

Custom reverse-mode (VJP) differentiation examples.

.. automodule:: autodiff.custom_vjp
   :members:
   :undoc-members:
   :show-inheritance:

**Key examples**:

* ``custom_sqrt`` / ``custom_sqrt_vjp`` – custom VJP for ``sqrt(x)``.
* ``clip_gradient_vjp`` – straight-through estimator for gradient clipping.
* ``smooth_abs_vjp`` – differentiable absolute value with smooth backward pass.

autodiff.custom_jvp
-------------------

Custom forward-mode (JVP) differentiation examples.

.. automodule:: autodiff.custom_jvp
   :members:
   :undoc-members:
   :show-inheritance:

**Key examples**:

* ``leaky_relu_jvp`` – leaky ReLU with explicit JVP rule.
* ``smooth_abs_jvp`` – smooth absolute value with JVP.
* ``learnable_activation_jvp`` – parameterised activation with custom JVP.
