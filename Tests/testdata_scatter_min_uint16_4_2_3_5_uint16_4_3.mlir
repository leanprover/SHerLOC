"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x050002000300010001000100010000000100040001000100020004000000010002000100010002000100020003000000010000000100040001000200010000000600060000000100060001000000000002000100000007000100030002000400000001000200010002000100000000000400020004000100000002000200000001000100020002000600010001000300010001000500010002000100070001000000000002000300040000000400030001000000060001000300020004000100020000000000020000000000000000000300000003000000050002000100000001000500000000000200000002000300"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[3, 0, 2], [4, 5, 3], [2, 2, 4], [5, 3, 3]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui16>, tensor<4x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x050002000300010001000100010000000100000001000100020004000000010002000100010002000100020003000000010000000100040001000200010000000600060000000100060001000000000002000100000007000100030002000400000001000200010002000100000000000400020004000100000002000200000001000100020002000600010001000300010001000400010002000100070001000000000002000300040000000400030001000000060001000300020004000100020000000000020000000000000000000300000003000000050002000100000001000500000000000200000002000300"> : tensor<4x2x3x5xui16>}> : () -> tensor<4x2x3x5xui16>
    "func.return"(%0) : (tensor<4x2x3x5xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

