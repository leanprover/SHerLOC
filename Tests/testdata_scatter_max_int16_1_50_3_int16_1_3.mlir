"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi16>, tensor<1x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0000030003000300000003000100FBFF00000200FBFF00000000070003000000000005000000FAFF0400030002000200FEFF00000200FDFFFEFFFAFFFFFF00000300FCFFFFFF000000000100FDFF01000400FDFF0300FFFF0100FFFFFFFF03000300FFFF00000000020001000100FCFFFCFF00000400FEFF0100FDFF0000FFFF0000FEFF0400FCFF0500FDFF0600020001000000FDFF03000100FDFF03000000FDFF04000000FBFF000000000000FDFFFDFFFDFF02000000010003000200000002000500FDFFFEFF00000100FFFF0200FEFFFEFFFFFFFCFF0100000000000300FDFFFFFF000001000000FDFFFFFFFFFF00000000FDFFFFFF0300000000000300FEFF0000FCFF05000100010000000100FDFF0100FBFF00000200FFFFFEFF030001000400FCFFFFFFFEFFFFFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, -2, 1]]> : tensor<1x3xi16>}> : () -> tensor<1x3xi16>
    "func.return"(%1, %2) : (tensor<1x50x3xi16>, tensor<1x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0000030003000300000003000100FBFF00000200FBFF00000000070003000000000005000000FAFF0400030002000200FEFF00000200FDFFFEFFFAFFFFFF00000300FCFFFFFF000000000100FDFF01000400FDFF0300FFFF0100FFFFFFFF03000300FFFF00000000020001000100FCFFFCFF00000400FEFF0100FDFF0000FFFF0000FEFF0400FCFF0500FDFF0600020001000000FDFF03000100FDFF03000000FDFF04000000FBFF000000000000FDFFFDFFFDFF020000000100030002000000020005000100FEFF00000100FFFF0200FEFFFEFFFFFFFCFF0100000000000300FDFFFFFF000001000000FDFFFFFFFFFF00000000FDFFFFFF0300000000000300FEFF0000FCFF05000100010000000100FDFF0100FBFF00000200FFFFFEFF030001000400FCFFFFFFFEFFFFFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    "func.return"(%0) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

