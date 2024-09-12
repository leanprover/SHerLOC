"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui16>, tensor<2x7xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      "stablehlo.return"(%arg1) : (tensor<ui16>) -> ()
    }) : (tensor<5x6x7xui16>, tensor<2x2xi64>, tensor<2x7xui16>) -> tensor<5x6x7xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui16>, tensor<2x7xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000002000100020000000700000000000600020003000300010001000300020005000000050000000100020003000400010000000100000000000000000006000200040001000100020002000100060006000200020002000300010001000100050002000100040000000700010000000000000001000000000002000300020000000100050001000200040002000000020001000300050004000200000000000000000000000000020000000100000000000300020003000100030001000200010000000200010003000000030003000300010003000100000000000200000001000400030005000100020000000000010001000200020001000200030000000100000002000000030001000000020005000300020001000300010003000200030001000500040000000000000001000000030004000500030004000100040003000100070006000700000003000200000000000000030001000000030002000100000001000100000002000200000000000000030003000300010001000000040001000300000000000100010002000400000002000300000005000100000000000200"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 1, 2, 0, 2, 6, 0], [0, 0, 0, 3, 3, 1, 3]]> : tensor<2x7xui16>}> : () -> tensor<2x7xui16>
    "func.return"(%1, %2) : (tensor<5x6x7xui16>, tensor<2x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000002000100020000000700000001000100020000000200060000000300020005000000050000000100020003000400010000000100000000000000000006000200040001000100020002000100060006000200020002000300010001000100050002000100040000000700010000000000000001000000000002000300020000000100050001000200040002000000020001000300050004000200000000000000000000000000020000000100000000000300020003000100030001000200010000000200010003000000030003000300000000000000030003000100030001000400030005000100020000000000010001000200020001000200030000000100000002000000030001000000020005000300020001000300010003000200030001000500040000000000000001000000030004000500030004000100040003000100070006000700000003000200000000000000030001000000030002000100000001000100000002000200000000000000030003000300010001000000040001000300000000000100010002000400000002000300000005000100000000000200"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    "func.return"(%0) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

