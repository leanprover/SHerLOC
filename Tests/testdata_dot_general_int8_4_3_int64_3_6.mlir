"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xi64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xi64>) -> tensor<3x6xi64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    "func.return"(%7) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[7, 6, 0], [0, 1, -1], [2, 1, -4], [3, 0, 1]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 1, -3, 7, 0, 0], [0, 0, 5, -5, -1, 0], [0, 5, 0, 4, 4, 1]]> : tensor<3x6xi64>}> : () -> tensor<3x6xi64>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[7, 7, 9, 19, -6, 0], [0, -5, 5, -9, -5, -1], [2, -18, -1, -7, -17, -4], [3, 8, -9, 25, 4, 1]]> : tensor<4x6xi64>}> : () -> tensor<4x6xi64>
    "func.return"(%0) : (tensor<4x6xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

