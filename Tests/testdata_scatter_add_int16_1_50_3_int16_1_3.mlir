"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi16>, tensor<1x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0200000007000500FFFFFDFFFFFFFFFF03000300000002000100FFFF0000FEFF000000000000FFFF0000FFFF0100F8FF00000400FEFF0000FEFF0000FEFFFBFF00000000030004000100FCFF02000500FFFF0000FAFF020003000400010006000300FEFFFFFF090000000100FCFF02000000FFFF020000000100FDFFFFFFFBFF000001000000FCFFFAFFFCFF010004000200FEFF00000100FDFF00000000FEFF000003000600FEFFFFFF0300000001000000000000000500050000000000FDFF0000000005000100FFFF0300FDFFFFFFFBFF0500FEFF01000000FCFF0500FCFFFAFF000000000500FFFF020002000100FFFF0100FFFF010000000500FFFF000001000000000002000000FCFFFFFF01000000FEFF000001000000FEFF03000300FDFF0200FBFF0100FFFF0300"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[2, -3, 0]]> : tensor<1x3xi16>}> : () -> tensor<1x3xi16>
    "func.return"(%1, %2) : (tensor<1x50x3xi16>, tensor<1x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0200000007000500FFFFFDFFFFFFFFFF03000300000002000100FFFF0000FEFF000000000000FFFF0000FFFF0100F8FF00000400FEFF0000FEFF0000FEFFFBFF00000000030004000100FCFF02000500FFFF0000FAFF020003000400010006000300FEFFFFFF090000000100FCFF02000000FFFF020000000100FDFFFFFFFBFF000001000000FCFFFAFFFCFF010004000200FEFF00000100FDFF00000000FEFF000003000600FEFFFFFF0300000001000000000000000500050000000000FDFF0200FDFF05000100FFFF0300FDFFFFFFFBFF0500FEFF01000000FCFF0500FCFFFAFF000000000500FFFF020002000100FFFF0100FFFF010000000500FFFF000001000000000002000000FCFFFFFF01000000FEFF000001000000FEFF03000300FDFF0200FBFF0100FFFF0300"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    "func.return"(%0) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

