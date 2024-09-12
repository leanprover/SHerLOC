"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui16>, tensor<5x2x2x7xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<5x6x7xui16>, tensor<2x2x1xi64>, tensor<5x2x2x7xui16>) -> tensor<5x6x7xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui16>, tensor<5x2x2x7xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x040004000000020000000100010000000100010002000000020003000100050001000100020001000100020000000100020002000100010000000000010004000000020000000000000001000300010002000100010001000100000003000000000003000000020000000100000000000200030007000400000004000000010001000300040000000000030000000200030000000100050001000000010003000200020000000500000002000100050001000300040000000000020002000100010004000400000005000400020001000500040001000500000004000600030004000100010003000500030004000200020001000000000000000000000001000000070002000300000001000300020002000100050001000200040004000100020000000200000007000200020001000000030000000000010001000000040002000200050001000100000004000100020000000100010002000200040005000200000002000300010003000100020003000000030001000100060001000200020001000000040003000200000000000100040004000100030004000000050004000200"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    %2 = "stablehlo.constant"() <{value = dense<"0x01000100040000000500000001000100060002000000020002000000000004000100050004000800000000000100030003000000070000000300040001000300010005000200010003000300010000000200030003000100020000000000010000000300000004000700000000000500000001000100010000000300020004000100010002000000020001000500040005000600000004000000010000000600020002000100010003000000000003000500020005000100010002000100020000000400010000000200000001000000010002000200000000000000050001000000020002000200000002000100020002000000040003000400030004000000010001000100020003000300010001000600010001000200"> : tensor<5x2x2x7xui16>}> : () -> tensor<5x2x2x7xui16>
    "func.return"(%1, %2) : (tensor<5x6x7xui16>, tensor<5x2x2x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010001000000000000000000010000000100010000000000020000000000040001000100020001000000000000000100020000000100000000000000010004000000020000000000000001000300010002000100010001000100000001000000000001000000020000000000000000000200010002000000000001000000010000000300040000000000030000000200030000000100050001000000010003000200020000000500000001000100010000000300020000000000010002000000010001000400000005000400000001000000010000000500000002000100010004000100010003000500030004000200020001000000000000000000000000000000030002000200000001000100020001000100000001000100000002000000010000000100000002000000000000000000010000000000010001000000040002000200050001000100000004000100000000000100010000000200010002000200000002000300010003000100000001000000010001000100030001000100020001000000020003000200000000000100040004000100030004000000050004000200"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    "func.return"(%0) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

