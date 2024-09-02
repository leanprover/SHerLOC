module @jit_plus23 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<f32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<4> : tensor<i32>
    %cst = stablehlo.constant dense<5.300000e+00> : tensor<f32>
    return %c, %cst : tensor<i32>, tensor<f32>
  }
}
