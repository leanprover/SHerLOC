module @jit_plus18 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<complex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(1.000000e+00,1.000000e+00)> : tensor<complex<f32>>
    return %cst : tensor<complex<f32>>
  }
}
