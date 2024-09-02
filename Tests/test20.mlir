module @jit_plus21 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
    return %cst : tensor<4x4xf32>
  }
}
