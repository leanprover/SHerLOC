"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xui8>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi8>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xi8>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui8>) -> tensor<3x6xi8>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi8>, tensor<3x6xi8>) -> tensor<4x6xi8>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi8>, tensor<4x6xi8>) -> ()
    "func.return"(%7) : (tensor<4x6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 9, 0], [3, -1, 2], [-4, -5, -4], [-1, 2, -1]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 0, 4, 2, 0, 3], [0, 6, 6, 3, 0, 2], [0, 2, 0, 1, 1, 0]]> : tensor<3x6xui8>}> : () -> tensor<3x6xui8>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 54, 54, 27, 0, 18], [3, -2, 6, 5, 2, 7], [-4, -38, -46, -27, -4, -22], [-1, 10, 8, 3, -1, 1]]> : tensor<4x6xi8>}> : () -> tensor<4x6xi8>
    "func.return"(%0) : (tensor<4x6xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

