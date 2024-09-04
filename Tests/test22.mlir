module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]> : tensor<4x3xi32>
    return %c : tensor<4x3xi32>
  }
}
