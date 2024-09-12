"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x4xi1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<i1>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6x4xi1>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 2, 1>, edge_padding_low = array<i64: 1, 0>, interior_padding = array<i64: 1, 0>}> : (tensor<2x3xi1>, tensor<i1>) -> tensor<6x4xi1>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6x4xi1>, tensor<6x4xi1>) -> ()
    "func.return"(%5) : (tensor<6x4xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<i1>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    "func.return"(%1, %2) : (tensor<2x3xi1>, tensor<i1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x4xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[false, false, false, false], [true, true, true, false], [false, false, false, false], [true, true, true, false], [false, false, false, false], [false, false, false, false]]> : tensor<6x4xi1>}> : () -> tensor<6x4xi1>
    "func.return"(%0) : (tensor<6x4xi1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

