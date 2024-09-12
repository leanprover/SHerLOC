"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui32>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui32>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui32>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 0, 3], [0, 2, 0], [3, 0, 0], [3, 3, 5]]> : tensor<4x3xui32>}> : () -> tensor<4x3xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[4.15925264, 3.60511088, 1.3781395, 1.36220229, 5.20291901, 1.18606138], [2.79948449, -6.24050951, 1.52686501, -3.68102241, 0.540223777, -1.89685023], [5.18335104, -2.77697968, 3.13168025, 0.119321391, -2.42349887, -1.81318474]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xui32>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[23.8685589, -1.12071729, 12.1513195, 3.08236885, 3.13534141, -3.06743145], [5.59896898, -12.481019, 3.053730e+00, -7.36204481, 1.08044755, -3.79370046], [12.4777584, 10.8153324, 4.13441849, 4.08660698, 15.608757, 3.55818415], [46.7929649, -21.7910938, 24.373415, -6.35985327, 5.11193466, -11.1982899]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

