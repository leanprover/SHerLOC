module @jit_plus attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.layout_mode = "default"}, %arg1: tensor<i32> {mhlo.layout_mode = "default"}) -> (tensor<i32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}
