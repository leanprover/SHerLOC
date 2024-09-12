"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xui64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xui64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui64>) -> tensor<3x6xui64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    "func.return"(%7) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, -1, -4], [-3, -1, -1], [1, 0, 0], [1, 4, 5]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 1, 5, 1, 0], [3, 2, 2, 1, 0, 2], [2, 2, 0, 3, 1, 3]]> : tensor<3x6xui64>}> : () -> tensor<3x6xui64>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[18446744073709551605, 18446744073709551610, 2, 7, 0, 18446744073709551602], [18446744073709551611, 18446744073709551609, 18446744073709551611, 18446744073709551597, 18446744073709551612, 18446744073709551611], [0, 1, 1, 5, 1, 0], [22, 19, 9, 24, 6, 23]]> : tensor<4x6xui64>}> : () -> tensor<4x6xui64>
    "func.return"(%0) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

