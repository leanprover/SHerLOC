"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x50x3xui16>, tensor<1xi64>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui16>, tensor<1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x050003000300040001000400000004000800040006000100000000000100020007000000070001000300000004000100050000000000040004000000000002000000000003000600020004000500000005000000010005000500010003000200060000000100040001000000000002000200030001000100000000000300000006000600010004000400030003000100000001000500050001000200010001000000040001000200040001000300020000000100010001000500000002000300040000000400010004000100030000000300040002000300010004000000040001000000010001000000040003000000000000000000000002000200000000000100010005000300000001000100030001000300010002000200030001000100040001000200030002000200"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 0, 0]]> : tensor<1x3xui16>}> : () -> tensor<1x3xui16>
    "func.return"(%1, %2) : (tensor<1x50x3xui16>, tensor<1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x050003000300040001000400000004000800040006000100000000000100020007000000070001000300000004000100050000000000040004000000000002000000000003000600020004000500000005000000010005000500010003000200060000000100040001000000000002000200030001000100000000000300000006000600010004000400030003000100000001000500050001000200010001000000040001000200040001000300020000000100010001000500000002000300100000000000010004000100030000000300040002000300010004000000040001000000010001000000040003000000000000000000000002000200000000000100010005000300000001000100030001000300010002000200030001000100040001000200030002000200"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    "func.return"(%0) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

