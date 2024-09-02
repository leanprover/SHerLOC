module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %0 = call @_cumulative_reduction(%cst) : (tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @_cumulative_reduction(%arg0: tensor<3xf32> {mhlo.layout_mode = "default"}) -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %0 = call @cumsum(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
  func.func private @cumsum(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
