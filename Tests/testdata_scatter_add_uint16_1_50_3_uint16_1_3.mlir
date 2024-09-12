"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x50x3xui16>, tensor<1xi64>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui16>, tensor<1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000001000200070000000300000001000400010002000100000000000600040000000300000004000400030004000300030004000000010001000400050002000100000006000100000002000200000000000100000000000000020001000100020001000200020007000500020002000100000006000000000000000100030004000300010003000000020001000200010003000000050000000400030001000100030001000200010000000000030000000200010001000100060003000200040003000000050001000000000002000000030002000300030003000300060002000400010001000300020001000000040006000000020001000500030003000000020006000000070005000300020000000100000003000200010003000200030000000100050002000000"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 4, 2]]> : tensor<1x3xui16>}> : () -> tensor<1x3xui16>
    "func.return"(%1, %2) : (tensor<1x50x3xui16>, tensor<1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000001000200070000000300000001000400010002000100000000000600040000000300000004000400030004000300030004000000010001000400050002000100000006000100000002000200000000000100000000000000020001000100020001000200020007000500020002000100000006000000000000000100030004000300010003000000020001000200010003000000050000000400030001000100030001000200010000000000030000000200010001000100060003000200080007000200050001000000000002000000030002000300030003000300060002000400010001000300020001000000040006000000020001000500030003000000020006000000070005000300020000000100000003000200010003000200030000000100050002000000"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    "func.return"(%0) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

