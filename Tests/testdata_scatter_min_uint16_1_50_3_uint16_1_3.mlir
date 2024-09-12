"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui16>, tensor<1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<1x50x3xui16>, tensor<1xi64>, tensor<1x3xui16>) -> tensor<1x50x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui16>, tensor<1x50x3xui16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui16>, tensor<1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x010004000100040001000000050002000000030002000000000000000100030000000000020001000200020000000600000001000500010001000200040000000100010000000100010005000300010005000400040001000100010001000000000003000300000001000300010002000000030004000400040000000100040000000000030002000400000000000700000002000100000001000100030001000400030000000300020004000200020001000000020003000300060002000400000002000100040004000000020000000100000001000000030000000400010000000400080000000200000001000000000000000200010001000000000000000000010000000200010000000300010001000100010000000000060001000300000002000300040000000600"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 5, 0]]> : tensor<1x3xui16>}> : () -> tensor<1x3xui16>
    "func.return"(%1, %2) : (tensor<1x50x3xui16>, tensor<1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010004000100040001000000050002000000030002000000000000000100030000000000020001000200020000000600000001000500010001000200040000000100010000000100010005000300010005000400040001000100010001000000000003000300000001000300010002000000030004000400040000000100040000000000030002000400000000000700000002000100000001000100030001000400030000000300020004000200020001000000020003000300060002000400000002000000040004000000020000000100000001000000030000000400010000000400080000000200000001000000000000000200010001000000000000000000010000000200010000000300010001000100010000000000060001000300000002000300040000000600"> : tensor<1x50x3xui16>}> : () -> tensor<1x50x3xui16>
    "func.return"(%0) : (tensor<1x50x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

