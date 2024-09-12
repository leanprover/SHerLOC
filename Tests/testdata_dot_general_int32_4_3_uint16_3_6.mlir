"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xui16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xi32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui16>) -> tensor<3x6xi32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi32>, tensor<3x6xi32>) -> tensor<4x6xi32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
    "func.return"(%7) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 3, -2], [5, 3, 0], [0, -3, 0], [-1, 1, 1]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 1, 0, 5, 2, 2], [3, 0, 0, 2, 3, 1], [1, 2, 4, 0, 2, 4]]> : tensor<3x6xui16>}> : () -> tensor<3x6xui16>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[7, -4, -8, 6, 5, -5], [19, 5, 0, 31, 19, 13], [-9, 0, 0, -6, -9, -3], [2, 1, 4, -3, 3, 3]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%0) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

