"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2xi64>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000000000600FEFF03000000FCFF0100FEFFFEFFFCFFFEFFFEFFFEFFFFFFFEFF01000600FEFF0500FFFF0100FEFF030004000200FFFFFAFF06000200000001000000040000000400FBFFFEFFFEFFFEFF02000300FBFF0000000003000000FFFF0300030006000200FFFF00000300FEFFFEFF0000FEFFFEFF0400020003000000010002000100FCFF0300000002000100FDFFFFFF04000400010000000300010001000100000004000300000000000200FEFFFFFF0300050000000000FEFFFDFF05000600FFFF05000000FEFF0300FFFF05000200000000000500FEFFFCFFFCFF01000800FEFFFCFF0400FFFFFDFFFFFFFCFFFFFF0000FFFFFEFF0000FDFFFCFF00000200FDFFFFFF0500FCFFFDFF000004000000020002000100FCFFFEFF0100010003000300030002000700FDFF01000600FEFF000003000000010002000400020000000200010003000400FFFF010003000000FFFFFFFFFEFF010003000000000001000400FCFF0300FFFFFDFF0200010002000000000003000100050001000100FDFFFFFFFFFF0200FEFFFEFF0100FEFFFEFFFAFF0300010000000300FDFF00000000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-4, -1, -4, -4, 1, 2, 1], [-3, 2, 0, -1, -1, -1, 1]]> : tensor<2x7xi16>}> : () -> tensor<2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000000000600FEFF03000000FCFFFDFFFDFFFAFFF8FFFFFF0000FFFFFFFFFEFF01000600FEFF0500FFFF0100FEFF030004000200FFFFFAFF06000200000001000000040000000400FBFFFEFFFEFFFEFF02000300FBFF0000000003000000FFFF0300030006000200FFFF00000300FEFFFEFF0000FEFFFEFF0400020003000000010002000100FCFF0300000002000100FDFFFFFF04000400010000000300010001000100000004000300000000000200FEFFFFFF0300050000000000FEFFFDFF05000600FFFF05000000FEFF0300FFFF0500FFFF020000000400FDFFFBFFFDFF01000800FEFFFCFF0400FFFFFDFFFFFFFCFFFFFF0000FFFFFEFF0000FDFFFCFF00000200FDFFFFFF0500FCFFFDFF000004000000020002000100FCFFFEFF0100010003000300030002000700FDFF01000600FEFF000003000000010002000400020000000200010003000400FFFF010003000000FFFFFFFFFEFF010003000000000001000400FCFF0300FFFFFDFF0200010002000000000003000100050001000100FDFFFFFFFFFF0200FEFFFEFF0100FEFFFEFFFAFF0300010000000300FDFF00000000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

