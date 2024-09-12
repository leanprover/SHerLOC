"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xui32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xui32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xui32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui32>) -> tensor<3x6xui32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xui32>, tensor<3x6xui32>) -> tensor<4x6xui32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xui32>, tensor<4x6xui32>) -> ()
    "func.return"(%7) : (tensor<4x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 3, 0], [1, 1, 1], [2, 2, 3], [0, 1, 0]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 3, 3, 5, 0], [2, 3, 4, 0, 1, 2], [4, 1, 1, 0, 3, 1]]> : tensor<3x6xui32>}> : () -> tensor<3x6xui32>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[6, 11, 18, 6, 13, 6], [6, 5, 8, 3, 9, 3], [16, 11, 17, 6, 21, 7], [2, 3, 4, 0, 1, 2]]> : tensor<4x6xui32>}> : () -> tensor<4x6xui32>
    "func.return"(%0) : (tensor<4x6xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

