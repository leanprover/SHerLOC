"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      "stablehlo.return"(%arg1) : (tensor<ui16>) -> ()
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    "func.return"(%6) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui16>, tensor<1xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x00000300040005000000030001000300030001000200050003000700020001000200020002000300020002000200040002000400030001000000050002000600000002000100020001000600000005000200040001000200000001000200040002000500050000000100020002000000010000000500040003000400050001000400000005000200030001000600000002000000010000000100000000000100050000000300010003000100010002000500040002000200010000000200040000000300000000000100010001000200040000000400030000000000060001000500000004000100050000000200020000000100030001000100"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xui16>}> : () -> tensor<1xui16>
    "func.return"(%1, %2) : (tensor<1x125xui16>, tensor<1xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02000300040005000000030001000300030001000200050003000700020001000200020002000300020002000200040002000400030001000000050002000600000002000100020001000600000005000200040001000200000001000200040002000500050000000100020002000000010000000500040003000400050001000400000005000200030001000600000002000000010000000100000000000100050000000300010003000100010002000500040002000200010000000200040000000300000000000100010001000200040000000400030000000000060001000500000004000100050000000200020000000100030001000100"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    "func.return"(%0) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

