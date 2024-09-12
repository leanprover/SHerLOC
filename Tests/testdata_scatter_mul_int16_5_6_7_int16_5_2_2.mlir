"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x2xi64>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02000100010002000000000005000000FFFFFDFF03000000FFFFFDFFFCFFFEFF030000000000FBFF0100FEFFFEFFFCFF0600FDFFFEFF000001000000FCFF00000100040001000100FFFF03000100F9FF0000010000000100FEFFFAFF0500FCFF03000600FEFF0200FDFF010002000300FFFF030000000400040000000400FCFF040003000600FEFF0600FEFF0500FDFFFFFF00000100000000000100FDFF02000000010000000000020001000400030001000100FEFF0000FBFF030000000300010000000000FFFF0300FDFF02000000FFFF0300FEFF08000300FEFF0000010001000100010001000000FDFF01000200FFFFFBFF01000000FFFFFCFF0300F9FFFCFFFFFFFDFF0100FFFF0000010000000100FEFF0100FEFFFFFF00000200020006000000000003000200000001000200FFFFFCFF0400FFFF0400FFFFFCFF000001000300FCFFFEFF0100FDFFFFFF00000000FFFF040000000000000000000200FDFF02000200FCFF020002000400FEFF0600000000000600FCFFFEFF01000400FFFF020005000200FFFF0000FBFF04000600FEFF0000010002000400FEFF0000FEFF0000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[[3, -1], [0, 2]], [[1, -1], [0, 0]], [[-3, 1], [2, 0]], [[-4, 0], [0, -1]], [[1, -3], [0, 3]]]> : tensor<5x2x2xi16>}> : () -> tensor<5x2x2xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02000300010002000000000005000000FFFFFAFF03000000FFFFFDFFFCFFFEFF030000000000FBFF0100FEFFFEFFFCFF0600FDFFFEFF000000000000FCFF00000100040001000100FFFF03000100F9FF0000010000000100FEFFFAFF0500FCFF03000600FEFF0000FDFF010002000300FFFF03000000FCFF040000000400FCFF040003000600FEFF0600FEFF0000FDFFFFFF00000100000000000100FDFF020000000100000000000200FDFF0400030001000100FEFF0000FBFF000000000300010000000000FFFF0300FDFF02000000FFFF0300FEFF08000300FEFF0000010002000100010001000000FDFF01000200FFFFFBFF01000000FFFFFCFF03001C00FCFFFFFFFDFF0100FFFF0000010000000100FEFF0100FEFFFFFF00000200000006000000000003000200000001000200FFFFFCFF0000FFFF0400FFFFFCFF000001000300FCFFFEFF0100FDFFFFFF00000000FFFF040000000000000000000200FDFF06000200FCFF020002000400FEFF0600000000000600FCFFFEFF01000400FFFF02000500020000000000FBFF04000600FEFF0000010002000400FEFF0000FEFF0000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

