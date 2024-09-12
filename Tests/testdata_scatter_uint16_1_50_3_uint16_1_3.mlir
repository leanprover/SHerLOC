"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      "stablehlo.return"(%arg1) : (tensor<ui16>) -> ()
    }) : (tensor<1x50x3xui16>, tensor<1xi64>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui16>, tensor<1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x020001000300020005000500020007000100030001000100030004000200000001000300030002000600000003000000010000000100040002000600050006000100010003000300020003000100050000000000000000000400000003000100010001000000020001000000010008000000020006000200010000000000000002000200010004000000010002000000030001000300000001000100010003000300020003000400020000000800000007000100040001000000000000000000000000000000000002000100000002000500040000000300010002000500030001000100010000000200040004000100000001000200010004000500020001000000030000000100000001000100000002000000000000000000030003000400020003000100030005000100"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 0, 1]]> : tensor<1x3xui16>}> : () -> tensor<1x3xui16>
    "func.return"(%1, %2) : (tensor<1x50x3xui16>, tensor<1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x020001000300020005000500020007000100030001000100030004000200000001000300030002000600000003000000010000000100040002000600050006000100010003000300020003000100050000000000000000000400000003000100010001000000020001000000010008000000020006000200010000000000000002000200010004000000010002000000030001000300000001000100010003000300020003000400020000000800000007000100040001000000000000000000040000000100000002000100000002000500040000000300010002000500030001000100010000000200040004000100000001000200010004000500020001000000030000000100000001000100000002000000000000000000030003000400020003000100030005000100"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    "func.return"(%0) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

