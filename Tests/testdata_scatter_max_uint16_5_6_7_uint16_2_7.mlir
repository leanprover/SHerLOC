"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui16>, tensor<2x7xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<5x6x7xui16>, tensor<2x2xi64>, tensor<2x7xui16>) -> tensor<5x6x7xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui16>, tensor<2x7xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000006000400000000000100030000000100020004000200030003000200000000000000030000000000000002000400040001000300010004000600020001000100000000000300030002000000020000000100010004000200030004000500050003000000020001000300010007000600000001000000030001000000030002000000010001000500020004000100000000000200020003000400020000000500000004000100040001000500000001000100020002000300010000000500040004000400000000000100000001000300050003000500000000000100000000000200030003000200010001000100000000000000000003000100020000000100010007000200020000000300000002000200010001000100020000000500000001000200000001000400020001000000000003000100040000000000020001000300050002000300020005000300020003000100030005000100010003000400000006000000000002000200000002000000020000000200040003000500000002000000010000000500040003000200020002000200050000000600040001000000"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[6, 3, 0, 2, 1, 1, 2], [4, 4, 2, 0, 5, 3, 3]]> : tensor<2x7xui16>}> : () -> tensor<2x7xui16>
    "func.return"(%1, %2) : (tensor<5x6x7xui16>, tensor<2x7xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000006000400000000000100030006000300020004000200030003000200000000000000030000000000000002000400040001000300010004000600020001000100000000000300030002000000020000000100010004000200030004000500050003000000020001000300010007000600000001000000030001000000030002000000010001000500020004000100000000000200020003000400020000000500000004000100040001000500000001000100020002000300010000000500040004000400000000000100000001000300050004000500000005000300030000000200030003000200010001000100000000000000000003000100020000000100010007000200020000000300000002000200010001000100020000000500000001000200000001000400020001000000000003000100040000000000020001000300050002000300020005000300020003000100030005000100010003000400000006000000000002000200000002000000020000000200040003000500000002000000010000000500040003000200020002000200050000000600040001000000"> : tensor<5x6x7xui16>}> : () -> tensor<5x6x7xui16>
    "func.return"(%0) : (tensor<5x6x7xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

