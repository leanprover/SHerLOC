module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %0 = stablehlo.sqrt %cst : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
