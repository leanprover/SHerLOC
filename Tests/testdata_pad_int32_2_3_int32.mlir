"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x7xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<i32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x7xi32>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 1, 2>, edge_padding_low = array<i64: 1, 2>, interior_padding = array<i64: 0, 0>}> : (tensor<2x3xi32>, tensor<i32>) -> tensor<4x7xi32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x7xi32>, tensor<4x7xi32>) -> ()
    "func.return"(%5) : (tensor<4x7xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<i32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    "func.return"(%1, %2) : (tensor<2x3xi32>, tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x7xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<4x7xi32>}> : () -> tensor<4x7xi32>
    "func.return"(%0) : (tensor<4x7xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

