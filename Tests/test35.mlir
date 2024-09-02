module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<9xui32> {mhlo.layout_mode = "default"}) -> (tensor<3x3xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<9xui32>) -> tensor<3x3xui32>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<3x3xui32>) -> tensor<3x3xui32>
    return %1 : tensor<3x3xui32>
  }
}
