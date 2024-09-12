"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xui64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xui64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui64>) -> tensor<3x6xui64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    "func.return"(%7) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, 1, 1], [2, 1, 0], [0, 2, 0], [8, 4, 5]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 0, 5, 4], [1, 0, 1, 2, 0, 0], [5, 0, 1, 1, 3, 1]]> : tensor<3x6xui64>}> : () -> tensor<3x6xui64>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[6, 0, 2, 3, 23, 17], [1, 0, 1, 2, 10, 8], [2, 0, 2, 4, 0, 0], [29, 0, 9, 13, 55, 37]]> : tensor<4x6xui64>}> : () -> tensor<4x6xui64>
    "func.return"(%0) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

