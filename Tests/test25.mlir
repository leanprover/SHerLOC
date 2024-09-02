module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
    %c_0 = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
    %0 = stablehlo.multiply %c, %c_0 : tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}
