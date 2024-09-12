"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xui64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xui64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui64>) -> tensor<3x6xui64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    "func.return"(%7) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 3, -5], [3, -1, 1], [-1, 0, 3], [5, 0, 3]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 8, 0, 0, 2], [0, 0, 5, 1, 6, 3], [0, 0, 3, 3, 0, 6]]> : tensor<3x6xui64>}> : () -> tensor<3x6xui64>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 18446744073709551604, 18, 18446744073709551595], [0, 0, 22, 2, 18446744073709551610, 9], [0, 0, 1, 9, 0, 16], [0, 0, 49, 9, 0, 28]]> : tensor<4x6xui64>}> : () -> tensor<4x6xui64>
    "func.return"(%0) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

