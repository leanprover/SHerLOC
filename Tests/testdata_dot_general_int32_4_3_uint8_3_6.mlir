"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xui8>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xi32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui8>) -> tensor<3x6xi32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi32>, tensor<3x6xi32>) -> tensor<4x6xi32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
    "func.return"(%7) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1, -2, -4], [0, -2, -2], [0, 0, 0], [3, -1, 4]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 5, 0, 1, 6, 1], [4, 1, 3, 5, 0, 4], [3, 2, 0, 4, 3, 1]]> : tensor<3x6xui8>}> : () -> tensor<3x6xui8>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-24, -15, -6, -27, -18, -13], [-14, -6, -6, -18, -6, -10], [0, 0, 0, 0, 0, 0], [20, 22, -3, 14, 30, 3]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%0) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

