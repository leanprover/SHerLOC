"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xui16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xui16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xui16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xui16>) -> tensor<3x6xui16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xui16>, tensor<3x6xui16>) -> tensor<4x6xui16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x6xui16>, tensor<4x6xui16>) -> ()
    "func.return"(%7) : (tensor<4x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2, -2, -4], [0, -2, -3], [-4, 0, 0], [0, -1, 3]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 5, 1, 0, 2, 2], [2, 2, 1, 3, 0, 0], [0, 0, 1, 0, 1, 2]]> : tensor<3x6xui16>}> : () -> tensor<3x6xui16>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[65532, 65522, 65528, 65530, 65528, 65524], [65532, 65532, 65531, 65530, 65533, 65530], [0, 65516, 65532, 0, 65528, 65528], [65534, 65534, 2, 65533, 3, 6]]> : tensor<4x6xui16>}> : () -> tensor<4x6xui16>
    "func.return"(%0) : (tensor<4x6xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

