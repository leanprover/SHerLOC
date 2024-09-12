"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      "stablehlo.return"(%arg1) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2xi64>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0000FCFFFEFF0300FFFFFEFF01000200F6FF00000100010000000600FEFF0000FFFF02000500FFFF0200FCFFFCFFFAFFFEFF0000FEFF0000000000000000FFFF01000000FDFF0400FFFF0000FBFFFFFFFFFF06000200FEFF0300FEFF05000500FCFF02000200FDFFFCFFFAFF0300FEFF0000FEFF04000600FFFF040002000000FFFF0000FCFF0000FFFF00000100FBFF0100FBFF03000300FFFF05000000FEFF02000000FDFF0200000000000400FBFF0100FDFFFFFF0000000005000200FEFF0000FAFFFFFF02000100060003000300F9FFFDFF010001000000FDFF0000F8FF02000000FAFF000002000300020000000000FAFF0700030002000200020001000000FBFFFDFFFEFF0200FEFF04000100040002000200FEFF0100FFFF0100020000000400FEFFFFFFFEFFFAFF04000000FDFF0100010002000100FEFF000001000000FBFF0400FAFF0200FEFF0000050000000000FEFFFEFF000002000400FBFFFAFF05000000FFFFFBFF0000FEFF0000000006000300FEFFFBFF0000FBFFFEFFFBFF0500FDFF05000000FEFFFEFF01000000000003000000060005000000F9FF01000000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 0, 1, 0, 4, -6, -5], [1, -2, 6, 0, 1, 1, 5]]> : tensor<2x7xi16>}> : () -> tensor<2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0000FCFFFEFF0300FFFFFEFF010002000000010000000400FAFFFBFFFEFF0000FFFF02000500FFFF0200FCFFFCFFFAFFFEFF0000FEFF0000000000000000FFFF01000000FDFF0400FFFF0000FBFFFFFFFFFF06000200FEFF0300FEFF05000500FCFF02000200FDFFFCFFFAFF0300FEFF0000FEFF04000600FFFF040002000000FFFF0000FCFF0000FFFF00000100FBFF0100FBFF03000300FFFF05000000FEFF02000000FDFF0200000000000400FBFF0100FDFFFFFF0000000005000200FEFF0000FAFFFFFF02000100060003000300F9FF0100FEFF0600000001000100050002000000FAFF000002000300020000000000FAFF0700030002000200020001000000FBFFFDFFFEFF0200FEFF04000100040002000200FEFF0100FFFF0100020000000400FEFFFFFFFEFFFAFF04000000FDFF0100010002000100FEFF000001000000FBFF0400FAFF0200FEFF0000050000000000FEFFFEFF000002000400FBFFFAFF05000000FFFFFBFF0000FEFF0000000006000300FEFFFBFF0000FBFFFEFFFBFF0500FDFF05000000FEFFFEFF01000000000003000000060005000000F9FF01000000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

