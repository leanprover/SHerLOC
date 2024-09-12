"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x4xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xui8>, tensor<ui8>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6x4xui8>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 2, 1>, edge_padding_low = array<i64: 1, 0>, interior_padding = array<i64: 1, 0>}> : (tensor<2x3xui8>, tensor<ui8>) -> tensor<6x4xui8>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6x4xui8>, tensor<6x4xui8>) -> ()
    "func.return"(%5) : (tensor<6x4xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xui8>, tensor<ui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2x3xui8>}> : () -> tensor<2x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<ui8>}> : () -> tensor<ui8>
    "func.return"(%1, %2) : (tensor<2x3xui8>, tensor<ui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x4xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<6x4xui8>}> : () -> tensor<6x4xui8>
    "func.return"(%0) : (tensor<6x4xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

