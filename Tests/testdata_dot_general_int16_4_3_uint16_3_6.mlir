"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xui16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xi16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xi16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui16>) -> tensor<3x6xi16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xi16>, tensor<3x6xi16>) -> tensor<4x6xi16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xi16>, tensor<4x6xi16>) -> ()
    "func.return"(%7) : (tensor<4x6xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, 0], [2, -1, -4], [2, 0, 0], [-1, -3, 0]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 2, 3, 2, 4, 1], [5, 1, 0, 0, 4, 1], [2, 0, 2, 7, 1, 2]]> : tensor<3x6xui16>}> : () -> tensor<3x6xui16>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 0, 0, 0], [-13, 3, -2, -24, 0, -7], [0, 4, 6, 4, 8, 2], [-15, -5, -3, -2, -16, -4]]> : tensor<4x6xi16>}> : () -> tensor<4x6xi16>
    "func.return"(%0) : (tensor<4x6xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

