"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui64>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui64>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui64>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 2, 0], [0, 1, 1], [1, 1, 1], [1, 3, 2]]> : tensor<4x3xui64>}> : () -> tensor<4x3xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[2.69352365, -1.75565517, 3.64608979, -4.61772156, 0.969550549, 3.7079103], [-0.878219723, -1.76270247, 3.62539315, 3.14396954, -2.84121752, 1.35875285], [1.46522605, 7.14490747, 4.94334316, -4.47644567, -0.379561633, -3.70317078]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xui64>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.937084197, -5.281060e+00, 10.8968763, 1.67021751, -4.71288443, 6.42541599], [0.58700633, 5.38220501, 8.56873607, -1.33247614, -3.22077918, -2.34441805], [3.280530e+00, 3.62654972, 12.2148266, -5.9501977, -2.25122857, 1.36349249], [2.98931646, 7.24605227, 24.4089546, -4.1387043, -8.31322479, 0.377827168]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

