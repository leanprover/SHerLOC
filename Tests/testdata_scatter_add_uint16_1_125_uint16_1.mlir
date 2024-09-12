"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    "func.return"(%6) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui16>, tensor<1xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0700000002000700040003000200010005000200030001000000010001000600010002000200010000000000010001000900020002000200010001000100000002000400030000000500080000000000020000000200020002000200000000000300010000000200050001000A000100000001000000020000000200020000000300010006000000000003000100010002000000020003000400020000000000020000000700000003000000020003000100000005000100020000000200000002000400010000000000000000000300040002000200030003000100000000000100010004000100010004000100040004000000020005000300"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xui16>}> : () -> tensor<1xui16>
    "func.return"(%1, %2) : (tensor<1x125xui16>, tensor<1xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0700000002000700040003000200010005000200030001000000010001000600010002000200010000000000010001000900020002000200010001000100000002000400030000000500080000000000020000000200020002000200000000000300010000000200050001000A000100000001000000020000000200020000000300010006000000000003000100010002000000020003000400020000000000020000000700000003000000020003000100000005000100020000000200000002000400010000000000000000000300040002000200030003000100000000000100010004000100010004000100040004000000020005000300"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    "func.return"(%0) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

