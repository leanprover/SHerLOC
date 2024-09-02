module @jit_plus2 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 1, 2]> : tensor<3xui8>
    return %c : tensor<3xui8>
  }
}
