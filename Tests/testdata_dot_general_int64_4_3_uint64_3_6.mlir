"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi64>, tensor<3x6xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi64>) -> tensor<4x3xi64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui64>) -> tensor<3x6xi64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    "func.return"(%7) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi64>, tensor<3x6xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 3, -5], [3, -3, 7], [-3, 1, 0], [0, 5, -3]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[3, 3, 1, 0, 1, 2], [1, 2, 5, 2, 1, 2], [2, 6, 1, 2, 0, 3]]> : tensor<3x6xui64>}> : () -> tensor<3x6xui64>
    "func.return"(%1, %2) : (tensor<4x3xi64>, tensor<3x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-7, -24, 10, -4, 3, -9], [20, 45, -5, 8, 0, 21], [-8, -7, 2, 2, -2, -4], [-1, -8, 22, 4, 5, 1]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%0) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

