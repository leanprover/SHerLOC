"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x040002000000020002000200020000000100010003000300040000000100010004000200010001000200010003000100050003000200020004000000010004000100000001000100000002000600070003000100010002000200010004000100030003000200010002000000020002000500010000000400010001000000030000000000030001000100000000000200030000000200030001000300020003000300030004000300040001000000050003000000020003000300010002000300030004000000000005000100040000000200000001000000000000000400040002000200000000000200000004000200"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 5, 0], [2, 3, 3], [1, 1, 0], [2, 3, 2]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui16>, tensor<4x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0400020000000200040002000200000001000600030003000400000001000100040002000100010002000100030001000500030002000200040000000100040001000000030001000000020006000A0003000100010002000500010004000100030003000200010002000000020002000500010000000400010001000000030001000000030001000100010000000200030000000200030001000300020003000300030004000300040001000000050003000000020003000300010004000300030004000000030005000100040000000400000001000000000000000400040002000200000000000200000004000200"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    "func.return"(%0) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

