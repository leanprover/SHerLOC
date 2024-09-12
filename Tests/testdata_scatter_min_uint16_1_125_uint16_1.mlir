"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    "func.return"(%6) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui16>, tensor<1xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x04000100010003000200020003000000030000000000010001000500020002000200040000000100040003000000050002000300020000000200020005000100010000000000020001000100020001000200030004000400000003000800000002000300000003000500000002000300000002000700040000000100030002000000000001000100010001000300010006000200000001000100000001000200000000000000000003000000040000000100030003000000010000000100040006000200040005000100030002000500030002000400020001000000060009000000000004000100060000000000010005000100000001000000"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xui16>}> : () -> tensor<1xui16>
    "func.return"(%1, %2) : (tensor<1x125xui16>, tensor<1xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00000100010003000200020003000000030000000000010001000500020002000200040000000100040003000000050002000300020000000200020005000100010000000000020001000100020001000200030004000400000003000800000002000300000003000500000002000300000002000700040000000100030002000000000001000100010001000300010006000200000001000100000001000200000000000000000003000000040000000100030003000000010000000100040006000200040005000100030002000500030002000400020001000000060009000000000004000100060000000000010005000100000001000000"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    "func.return"(%0) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

