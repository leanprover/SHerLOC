"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[1.38732839, 2.89946914, -2.876290e-01], [6.79478884, 1.76005256, 0.305416048]], [[1.63096201, 4.79242373, -1.44652247], [-0.525784791, -0.114975639, -1.21797776]], [[2.14421749, -0.641117454, -1.57080615], [1.77755022, -4.11711407, 3.59519815]], [[-2.79984736, -1.47077167, 3.22873116], [-1.34054029, -6.97518253, 0.675602198]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[0.105449297, 3.63069248]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<4x2x3xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.38732839, 2.89946914, -2.876290e-01], [6.79478884, 1.76005256, 0.305416048]], [[1.63096201, 4.79242373, -1.44652247], [-0.525784791, -0.114975639, -1.21797776]], [[2.14421749, -0.641117454, -1.57080615], [1.77755022, -4.11711407, 3.59519815]], [[-2.79984736, -1.47077167, 3.22873116], [-1.34054029, -6.97518253, 3.63069248]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    "func.return"(%0) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

