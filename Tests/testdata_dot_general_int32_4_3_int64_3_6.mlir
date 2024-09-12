"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xi64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xi64>) -> tensor<3x6xi64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    "func.return"(%7) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, -5, 1], [-1, -7, 4], [6, -4, 0], [-2, -1, -1]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, -1, 1, 6, 0, 1], [0, -4, -6, 2, -4, -1], [3, -2, -1, 0, 0, 2]]> : tensor<3x6xi64>}> : () -> tensor<3x6xi64>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3, 18, 29, -10, 20, 7], [13, 21, 37, -20, 28, 14], [-6, 10, 30, 28, 16, 10], [-1, 8, 5, -14, 4, -3]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%0) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

