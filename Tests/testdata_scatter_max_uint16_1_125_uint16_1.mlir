"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    "func.return"(%6) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui16>, tensor<1xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01000700010001000700050001000200020001000000000001000100010002000000020002000100000003000200020001000100020000000500010003000200010001000100050000000100010002000400010000000100020004000100020001000100040000000100020002000200000002000500010000000300000003000600050001000500010003000200040001000100000001000000020002000300020000000300030001000500050004000100020000000200020001000300010003000000010001000100040003000200030005000000030004000500010002000200020005000200010001000200020000000000010000000100"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xui16>}> : () -> tensor<1xui16>
    "func.return"(%1, %2) : (tensor<1x125xui16>, tensor<1xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01000700010001000700050001000200020001000000000001000100010002000000020002000100000003000200020001000100020000000500010003000200010001000100050000000100010002000400010000000100020004000100020001000100040000000100020002000200000002000500010000000300000003000600050001000500010003000200040001000100000001000000020002000300020000000300030001000500050004000100020000000200020001000300010003000000010001000100040003000200030005000000030004000500010002000200020005000200010001000200020000000000010000000100"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    "func.return"(%0) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

