"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    "func.return"(%6) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui16>, tensor<1xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01000000050001000200020001000200070001000400000003000300000000000000000002000300070000000200020000000000010003000200020004000100000003000300000003000100020000000600050001000000020002000000070000000000020002000700030008000200050001000200040003000200040004000000030003000100010002000100030003000000010002000200000000000500000002000000000002000100040008000400070002000000030003000100020003000000000003000300020004000200000004000000050000000000010000000100000001000200020001000000040001000000000004000200"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xui16>}> : () -> tensor<1xui16>
    "func.return"(%1, %2) : (tensor<1x125xui16>, tensor<1xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02000000050001000200020001000200070001000400000003000300000000000000000002000300070000000200020000000000010003000200020004000100000003000300000003000100020000000600050001000000020002000000070000000000020002000700030008000200050001000200040003000200040004000000030003000100010002000100030003000000010002000200000000000500000002000000000002000100040008000400070002000000030003000100020003000000000003000300020004000200000004000000050000000000010000000100000001000200020001000000040001000000000004000200"> : tensor<1x125xui16>}> : () -> tensor<1x125xui16>
    "func.return"(%0) : (tensor<1x125xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

