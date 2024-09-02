module @jit_plus19 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    return %c : tensor<i32>
  }
}
