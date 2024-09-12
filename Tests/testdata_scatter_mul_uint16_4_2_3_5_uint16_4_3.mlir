"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x04000200010000000300020003000300010006000200050000000000010004000100000005000300010001000100050004000300000003000300000005000700020004000100020001000200010002000700020001000500010002000200000002000300020001000100010003000000010003000000040000000000020002000000000000000300050005000200030002000100020001000400020001000000060003000100010002000200020000000A000200040001000400040002000200010000000300010001000100050004000300050003000300050000000000010003000100010003000100060003000000"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 0, 2], [2, 6, 4], [1, 1, 1], [1, 1, 6]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui16>, tensor<4x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0400020001000000060002000300030001000000020005000000000002000400010000000500030001000100010005000400030000000300030000000500070002000400020002000100020001000C000700020001000500040002000200000002000300020001000100010003000000010003000000040000000000020002000000000000000300050005000200030002000100020001000400020001000000060003000100010002000200020000000A000200040001000400040002000200010000000300010001000100050004001200050003000300050000000000010003000100010003000100060003000000"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    "func.return"(%0) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

