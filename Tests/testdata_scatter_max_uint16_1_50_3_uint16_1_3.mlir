"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x50x3xui16>, tensor<1xi64>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui16>, tensor<1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x010000000500010003000400020003000400000000000100030003000000000001000000080000000400000002000100040002000300040004000000040000000000030000000200000000000100000002000300010005000100010000000000030000000400010000000100010001000200000000000000000005000200010003000300080004000500010000000500020003000300020003000200030001000300000002000600020001000000000000000100030001000200000000000200030003000300020004000100030001000200030000000100030001000500000001000100000001000100010003000000000005000000040004000000020006000800010000000100020000000400010000000400040001000000010008000100020002000300020000000000"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 3]]> : tensor<1x3xui16>}> : () -> tensor<1x3xui16>
    "func.return"(%1, %2) : (tensor<1x50x3xui16>, tensor<1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010000000500010003000400020003000400000000000100030003000000000001000000080000000400000002000100040002000300040004000000040000000000030000000200000000000100000002000300010005000100010000000000030000000400010000000100010001000200000000000000000005000200010003000300080004000500010000000500020003000300020003000200030001000300000002000600020001000000000000000100030001000200000000000200030003000300020004000100030001000200030000000100030001000500000001000100000001000100010003000000000005000000040004000000020006000800010000000100020000000400010000000400040001000000010008000100020002000300020000000000"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    "func.return"(%0) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

