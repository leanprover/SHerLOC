"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.49575233, 2.57913351, 3.17360067], [2.86800289, 0.764879227, 3.50482368]], [[-0.787247478, -2.3107903, -0.8217265], [-5.53171682, 2.28917766, -5.91439295]], [[-2.06035209, 1.88200915, -1.49790323], [7.83903265, -1.51898813, -1.68348587]], [[2.52399182, -5.52001381, 1.54348922], [-2.56950092, -4.03907061, -1.26045203]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[1.93206847, 1.86264634]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<4x2x3xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.49575233, 2.57913351, 3.17360067], [2.86800289, 0.764879227, 3.50482368]], [[-0.787247478, -2.3107903, -0.8217265], [-5.53171682, 2.28917766, -5.91439295]], [[-2.06035209, 1.88200915, -1.49790323], [7.83903265, -1.51898813, -1.68348587]], [[2.52399182, -5.52001381, 2.98212695], [-2.56950092, -4.03907061, -2.34777641]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    "func.return"(%0) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

