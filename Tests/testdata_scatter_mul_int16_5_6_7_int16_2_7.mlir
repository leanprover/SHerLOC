"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2xi64>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x03000000050002000100FEFFFEFFFFFF0000FDFF0500FEFFFFFFFDFFFDFF0000FFFFFDFFFBFFFDFFFFFF000000000000FCFF02000000FEFF0100FDFF00000200FEFF02000100040005000200000000000200FDFF040003000000000003000600FEFF04000500FBFF0200000000000400000000000300FDFFFDFFFFFF0100FEFFFFFFFFFF08000000FFFFFFFF03000000FFFF000000000100FFFF0500020002000000FEFF000002000000070004000200FDFFFBFF000000000100FFFF00000200FBFF0100FBFF0200FFFF0000000000000200010000000100020000000200FDFF0500F6FF01000200FAFF0100FFFFFDFF0000FCFF0000FDFFFFFF0400050000000200FEFF0400000000000700F9FF0000FEFF0200FEFFFFFF04000500000000000100FFFFFDFFFFFFFBFF000005000200030000000400FEFFFCFF00000000FEFFFFFF00000000FEFF00000500FEFF010003000000010001000100070002000000FFFF010000000000000000000000FEFF010003000100FFFF04000000020000000000FCFF0000FFFFFFFF0000FFFF0000000001000400FEFFFFFFFDFF0100FAFF02000200"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2, -4, 0, -2, 1, -1, -3], [0, 0, 1, 3, 0, -3, 4]]> : tensor<2x7xi16>}> : () -> tensor<2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x03000000050002000100FEFFFEFF020000000000F6FFFEFF01000900FDFF0000FFFFFDFFFBFFFDFFFFFF000000000000FCFF02000000FEFF0100FDFF00000200FEFF02000100040005000200000000000200FDFF040003000000000003000600FEFF04000500FBFF0200000000000400000000000300FDFFFDFFFFFF0100FEFFFFFFFFFF08000000FFFFFFFF03000000FFFF000000000100FFFF0500020002000000FEFF000002000000070004000200FDFFFBFF000000000100FFFF00000200FBFF0100FBFF0200FFFF000000000000020000000000010006000000FAFFF4FF0500F6FF01000200FAFF0100FFFFFDFF0000FCFF0000FDFFFFFF0400050000000200FEFF0400000000000700F9FF0000FEFF0200FEFFFFFF04000500000000000100FFFFFDFFFFFFFBFF000005000200030000000400FEFFFCFF00000000FEFFFFFF00000000FEFF00000500FEFF010003000000010001000100070002000000FFFF010000000000000000000000FEFF010003000100FFFF04000000020000000000FCFF0000FFFFFFFF0000FFFF0000000001000400FEFFFFFFFDFF0100FAFF02000200"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

