"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui32>, tensor<1xui32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<1x125xui32>, tensor<1xi64>, tensor<1xui32>) -> tensor<1x125xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui32>, tensor<1x125xui32>) -> ()
    "func.return"(%6) : (tensor<1x125xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui32>, tensor<1xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x020000000300000001000000000000000400000003000000030000000000000004000000010000000000000001000000000000000400000002000000040000000000000003000000020000000000000001000000010000000000000001000000000000000100000000000000050000000200000000000000040000000100000000000000030000000400000000000000010000000000000001000000020000000400000004000000030000000000000001000000020000000000000000000000030000000100000005000000030000000600000001000000000000000200000006000000030000000100000005000000000000000400000001000000030000000100000001000000010000000400000001000000000000000000000000000000000000000400000002000000050000000300000005000000040000000000000003000000050000000100000004000000000000000200000004000000000000000300000001000000010000000000000001000000020000000200000003000000030000000500000002000000020000000000000004000000000000000000000003000000010000000000000006000000000000000200000005000000000000000200000000000000020000000300000001000000040000000100000007000000000000000400000000000000050000000B000000"> : tensor<1x125xui32>}> : () -> tensor<1x125xui32>
    %2 = "stablehlo.constant"() <{value = dense<6> : tensor<1xui32>}> : () -> tensor<1xui32>
    "func.return"(%1, %2) : (tensor<1x125xui32>, tensor<1xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x060000000300000001000000000000000400000003000000030000000000000004000000010000000000000001000000000000000400000002000000040000000000000003000000020000000000000001000000010000000000000001000000000000000100000000000000050000000200000000000000040000000100000000000000030000000400000000000000010000000000000001000000020000000400000004000000030000000000000001000000020000000000000000000000030000000100000005000000030000000600000001000000000000000200000006000000030000000100000005000000000000000400000001000000030000000100000001000000010000000400000001000000000000000000000000000000000000000400000002000000050000000300000005000000040000000000000003000000050000000100000004000000000000000200000004000000000000000300000001000000010000000000000001000000020000000200000003000000030000000500000002000000020000000000000004000000000000000000000003000000010000000000000006000000000000000200000005000000000000000200000000000000020000000300000001000000040000000100000007000000000000000400000000000000050000000B000000"> : tensor<1x125xui32>}> : () -> tensor<1x125xui32>
    "func.return"(%0) : (tensor<1x125xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

