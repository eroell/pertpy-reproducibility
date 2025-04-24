from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import jax
import jax.numpy as jnp
import time
import numpy as np

# Define a wrapper function that handles the Sinkhorn solver correctly
def sink(a, b, cost_matrix, epsilon, min_iterations, max_iterations):
    """Compute optimal transport with Sinkhorn's solver"""
    # Create geometry
    geom = Geometry(cost_matrix=cost_matrix, epsilon=epsilon)
    
    # Create problem
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    
    # We need to instantiate the solver OUTSIDE the JIT-compiled region
    # or at least ensure max_iterations is handled as a static argument
    # This is a key part of the fix
    solver = sinkhorn.Sinkhorn(
        threshold=1e-3,
        inner_iterations=10,  # Setting explicitly to avoid dynamic calculation
        min_iterations=min_iterations,
        max_iterations=max_iterations
    )
    
    # Call the solver
    out = solver(prob)
    return out.reg_ot_cost

# Use a different approach to vmap and jit that avoids the tracing issue
# First jit the individual function
sink_jit = jax.jit(sink, static_argnums=(4, 5))

# Then create a batched version that applies over first dimension of a
def batch_sink(a_batch, b, cost_matrix, epsilon, min_iterations, max_iterations):
    results = []
    for a_single in a_batch:
        result = sink_jit(a_single, b, cost_matrix, epsilon, min_iterations, max_iterations)
        results.append(result)
    return jnp.stack(results)

# And a double-batched version
def batch_sink_2d(a_batch, b_batch, cost_matrix, epsilon, min_iterations, max_iterations):
    results = []
    for b_single in b_batch:
        row_results = batch_sink(a_batch, b_single, cost_matrix, epsilon, min_iterations, max_iterations)
        results.append(row_results)
    return jnp.stack(results)

# Generate test data
np.random.seed(42)
n_samples_a = 5
n_samples_b = 4
n_features = 50

# Create normalized histograms
hist_a = jnp.array(np.random.rand(n_samples_a, n_features))
hist_a = hist_a / hist_a.sum(axis=1, keepdims=True)

hist_b = jnp.array(np.random.rand(n_samples_b, n_features))
hist_b = hist_b / hist_b.sum(axis=1, keepdims=True)

# Create cost matrix
cost_matrix = jnp.array(np.random.rand(n_features, n_features))

# Run the batched version
print("Testing batched computation...")
start = time.time()
result = batch_sink_2d(hist_a, hist_b, cost_matrix, 1e-2, 100, 100)
end = time.time()
print(f"Time with batched computation: {end - start:.4f} seconds")

# Alternative approach: for more efficient computation, try using scan instead of explicit loops
from jax import lax

# Define a single-element computation
def compute_single(_, inputs):
    a, b, cost_matrix, epsilon, min_iter, max_iter = inputs
    return None, sink_jit(a, b, cost_matrix, epsilon, min_iter, max_iter)

# More efficient batched version using scan
def batch_sink_scan(a_batch, b, cost_matrix, epsilon, min_iterations, max_iterations):
    # Prepare inputs for each item in the batch
    inputs = (a_batch, jnp.repeat(b[None], a_batch.shape[0], axis=0), 
              jnp.repeat(cost_matrix[None], a_batch.shape[0], axis=0),
              jnp.repeat(jnp.array(epsilon)[None], a_batch.shape[0]),
              min_iterations, max_iterations)
    
    # Use scan to process each element
    _, results = lax.scan(compute_single, None, inputs)
    return results

print("\nTesting scan-based implementation...")
# Note: for a real implementation, you'd need to adapt this to handle the 2D case