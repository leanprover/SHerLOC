module @jit_plus6 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xi4> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi4>
    return %c : tensor<3xi4>
  }
}
