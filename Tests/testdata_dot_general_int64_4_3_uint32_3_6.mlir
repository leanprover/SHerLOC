"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi64>, tensor<3x6xui32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi64>) -> tensor<4x3xi64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui32>) -> tensor<3x6xi64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    "func.return"(%7) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi64>, tensor<3x6xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2, -5], [0, 0, 3], [-5, 0, -2], [0, 3, 2]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 0, 0, 4, 2], [0, 2, 0, 2, 3, 6], [2, 4, 2, 1, 0, 7]]> : tensor<3x6xui32>}> : () -> tensor<3x6xui32>
    "func.return"(%1, %2) : (tensor<4x3xi64>, tensor<3x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-10, -16, -10, -1, 6, -23], [6, 12, 6, 3, 0, 21], [-4, -13, -4, -2, -20, -24], [4, 14, 4, 8, 9, 32]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%0) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

