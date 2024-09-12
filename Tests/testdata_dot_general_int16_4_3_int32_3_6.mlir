"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xi32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xi32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xi32>) -> tensor<3x6xi32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi32>, tensor<3x6xi32>) -> tensor<4x6xi32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
    "func.return"(%7) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 3, 0], [-3, 1, 0], [2, -3, -4], [0, 0, 3]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-3, -1, -4, 1, 0, 0], [-4, -1, 0, 2, 0, 3], [0, -3, 0, 7, -5, 0]]> : tensor<3x6xi32>}> : () -> tensor<3x6xi32>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-12, -3, 0, 6, 0, 9], [5, 2, 12, -1, 0, 3], [6, 13, -8, -32, 20, -9], [0, -9, 0, 21, -15, 0]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%0) : (tensor<4x6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

