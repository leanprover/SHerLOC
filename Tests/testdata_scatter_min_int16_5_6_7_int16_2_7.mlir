"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2xi64>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01000300FFFFFEFF0700050004000400FFFFFEFF03000100FBFFFFFF01000100FFFFFEFFFFFF00000200FFFFFFFF00000100FEFFFBFFFFFF01000000FEFFFDFF000000000000FDFF05000200FFFFFFFF03000600FBFF0000000000000100FBFF0300F9FF06000300010002000000FFFF0000FFFF0100FEFF01000300000000000000FFFF0500FDFF03000100FEFFFEFFFEFF000006000000FFFF0300FFFF000001000000FBFF0000FCFF0100FFFF0000FFFF0000FAFF00000300FDFF020002000100FCFFFAFF0700FEFF000000000000030001000000FEFF00000400FDFFFFFF0000010005000400FCFF0000FDFFFDFF02000100FFFFFFFFFEFF00000300FEFFFAFFFEFFFEFF0400000000000100FEFFFFFFFDFF000003000000FEFF0200FEFFFEFF000001000000FFFF0000FDFFFDFFF8FFFFFF00000000FAFF00000700FFFF0000FFFFFFFF01000100FEFFFEFF000002000000000004000000FFFF03000300030002000100FDFF00000000FBFF0900FDFF0000FDFF060003000100000000000100FDFFFFFF0200000001000100FDFF01000000FEFFFEFF06000000FFFF02000200FEFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2, 0, -1, 2, 5, -1, -7], [2, 4, 4, 0, 0, -2, -3]]> : tensor<2x7xi16>}> : () -> tensor<2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01000300FFFFFEFF070005000400FEFFFFFFFEFF02000100FBFFF9FF01000100FFFFFEFFFFFF00000200FFFFFFFF00000100FEFFFBFFFFFF01000000FEFFFDFF000000000000FDFF05000200FFFFFFFF03000600FBFF0000000000000100FBFF0300F9FF06000300010002000000FFFF0000FFFF0100FEFF01000300000000000000FFFF0500FDFF03000100FEFFFEFFFEFF000006000000FFFF0300FFFF000001000000FBFF0000FCFF0100FFFF0000FFFF0000FAFF00000300FDFF020002000100FCFFFAFF0700FEFF000000000000030001000000FEFF00000000FDFFFDFF0000010005000400FCFF0000FDFFFDFF02000100FFFFFFFFFEFF00000300FEFFFAFFFEFFFEFF0400000000000100FEFFFFFFFDFF000003000000FEFF0200FEFFFEFF000001000000FFFF0000FDFFFDFFF8FFFFFF00000000FAFF00000700FFFF0000FFFFFFFF01000100FEFFFEFF000002000000000004000000FFFF03000300030002000100FDFF00000000FBFF0900FDFF0000FDFF060003000100000000000100FDFFFFFF0200000001000100FDFF01000000FEFFFEFF06000000FFFF02000200FEFF"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

