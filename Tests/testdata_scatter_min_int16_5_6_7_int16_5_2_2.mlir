"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x2xi64>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01000100010001000000FCFF0000FDFFFDFF00000000FBFF04000000FEFF050000000100FFFF0000000003000200010001000400000007000200030000000000FEFF06000500FCFFFDFF02000000F9FF01000000FEFF0100010003000200FFFFFAFFFAFF0000FFFFFFFFFCFF000003000200FCFF0200FDFF0000000000000000FEFF030002000000FEFFFAFF000000000400FDFFFEFF0100FCFFFCFF0300FDFF010000000000010000000000FFFF0200FEFF0000FDFF0100FFFFFFFF000000000200000000000000FEFFFBFFFEFF0500F9FFFFFFFFFFFDFFFFFF0000060000000100FCFFFBFFFFFF03000400FCFF0000FDFF0000FEFFFCFFFEFFFCFF01000300FEFF0100FEFF0300000000000000000000000200FFFF0600FBFF00000000FEFF0000FDFF040000000200FFFFFDFF0000FEFFFAFF02000600FBFF0000FCFFFEFF010006000000FEFF00000100000000000000FBFF0400020000000000FBFF01000400FEFFFEFFFEFF0400FEFF00000000FCFFFCFF0100FDFF03000100FEFF0000FDFFFFFF0200020000000200000006000100FEFFFDFF0100FEFFFBFFFEFF0000FFFFFFFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 2], [-2, 0]], [[2, 0], [-2, -3]], [[4, -3], [0, 1]], [[-3, 0], [3, 3]], [[0, -1], [-3, -1]]]> : tensor<5x2x2xi16>}> : () -> tensor<5x2x2xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01000000010001000000FCFF0000FDFFFDFF00000000FBFF04000000FEFF050000000100FFFF000000000300020001000100040000000700FEFF030000000000FEFF06000500FCFFFDFF02000000F9FF01000000FEFF0100010003000200FFFFFAFFFAFF0000FDFFFFFFFCFF000003000200FCFF0200FDFF0000000000000000FEFF030002000000FEFFFAFFFEFF00000400FDFFFEFF0100FCFFFCFF0300FDFF010000000000010000000000FFFF0200FEFF0000FDFF0100FFFFFFFF000000000200000000000000FEFFFBFFFEFF0500F9FFFFFFFFFFFDFFFFFF0000060000000000FCFFFBFFFFFF03000400FCFF0000FDFF0000FEFFFCFFFEFFFCFF0100FDFFFEFF0100FEFF0300000000000000000000000200FFFF0600FBFF00000000FEFF0000FDFF040000000200FFFFFDFF0000FEFFFAFF02000600FBFF0000FCFFFEFF010006000000FEFF00000100000000000000FBFF0400020000000000FBFF01000400FEFFFEFFFEFF0400FEFF00000000FCFFFCFF0100FDFF03000100FEFF0000FDFFFFFF02000200FDFF0200000006000100FEFFFDFF0100FEFFFBFFFEFF0000FFFFFFFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

