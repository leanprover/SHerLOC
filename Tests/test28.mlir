module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
    %c_0 = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi32>
    %0 = stablehlo.convert %c : (tensor<3xi32>) -> tensor<3xf32>
    %1 = stablehlo.convert %c_0 : (tensor<3xi32>) -> tensor<3xf32>
    %2 = stablehlo.divide %0, %1 : tensor<3xf32>
    return %2 : tensor<3xf32>
  }
}
