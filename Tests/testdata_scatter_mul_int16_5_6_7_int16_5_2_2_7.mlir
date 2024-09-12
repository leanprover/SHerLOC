"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x1xi64>, tensor<5x2x2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x010000000000000002000000FFFF0100F8FFF9FFFFFF0300FEFF04000200FFFF0100FFFF0100FEFFFFFF010005000000000004000000FFFFFDFF0700FFFF02000100FDFFFDFFFEFF03000700FFFF050002000200000000000100FFFF0300000004000300000000000100020002000100FFFF0600FEFF01000000FCFFFEFF00000100FDFF0800FEFF0000FBFF04000100000001000300020001000000FFFF0000FEFF010005000100FFFF00000300FFFFFFFF0100FFFFFCFF020000000200060000000000FFFFFCFF000002000000FEFF010003000200040000000000FEFFFCFFFFFFFFFFFFFF0000FCFFFFFFFFFFFFFFFDFF0300FBFFFFFF0500040000000000FFFFFEFF0000000003000000000000000100FFFF010003000300010000000100FEFF0200000004000200FFFF0400FBFFFFFFFAFFFDFF0000FCFF040003000300040001000100FCFF01000400FEFF0200FEFF02000000FEFF0100FFFFFFFFFFFFFEFFFEFF00000000FEFF04000000FEFF0400FFFFFCFF01000000000002000100FEFF0400FBFF0300FCFF00000000F7FF000004000400FEFF03000100FEFFFFFF0400FFFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<"0x0300010002000000FFFFFDFF0000FDFF010002000400000001000100010000000000FFFF0100FFFF0100FDFF000000000000FBFF02000000000000000100FEFF0200FDFFFDFFFCFF00000200FBFFFFFF04000100000000000000FFFF0100FEFFFEFFFEFF03000000FFFFFEFF0500FFFF000002000000020003000000010001000200FFFFFFFFFEFFFFFF02000000FFFFFCFF00000000FFFF0300FEFF00000000FFFF04000000F8FF00000100040003000000FDFFF9FF0400FFFF04000100FBFF00000600030001000300040002000400FEFFFCFF0200FDFFFEFF070000000400FDFF00000500F8FFFDFF00000100FFFFFFFF01000000FDFF000000000200FDFFFDFF000005000000FCFF0400FDFF01000000030002000000"> : tensor<5x2x2x7xi16>}> : () -> tensor<5x2x2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0300000000000000FEFF00000000FDFFF8FFF2FFFCFF0000FEFF0400020000000000010001000200FFFFFDFF000000000000ECFF00000000FDFF0700FFFF02000100FDFFFDFFFEFF03000700FFFF050002000200000000000100020006000000F4FFF4FF00000000FBFFFEFF08000100000000000000FFFF000008000400000003000000F8FF04000000050004000100000001000300020001000000FFFF0000FEFF010005000100000000000000FEFFFDFF0000FFFFFCFF04000000FEFFF4FF000000000000040000000000000002000300FAFF000000000000000000002000FFFFFFFFFFFF0000FCFFFFFFFFFFFFFFFDFF0300FBFFFFFF0500040000000000FCFFFAFF00000000EBFF00000000000001000500000012000900010000000400FCFF08000000F0FF04000300F8FFDDFF0000E8FFFDFF0000FCFF040003000300040001000100FCFF01000400FEFF02000600000000001000FDFF0000FFFF01000200FEFF000000000000000000000600F4FF0000ECFF000000000000FAFF010000000C00F6FF0000FCFF00000000F7FF000004000400FEFF03000100FEFFFFFF0400FFFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

