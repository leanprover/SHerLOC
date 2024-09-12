"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      "stablehlo.return"(%arg1) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x2xi64>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x05000000FEFFFDFF0200000002000000FEFF02000200030002000200FFFF04000000FCFF0100FAFFFFFF000000000100040003000100FEFF0000FFFF0400FEFF0000FFFFFFFF0000FDFF010002000100FDFFFCFF000000000200FDFFFDFF0000FFFFFCFF01000400F7FF010002000000FAFFFFFFFAFFFEFF01000300FBFFFDFF0000FDFF0200FAFF020001000400000000000100FBFF06000300000000000000FCFF0300FCFF02000200FFFF0000FDFF05000200030003000100050000000200000004000100FCFFFEFF0200030003000000000000000000FFFF0200000003000200FEFF00000100010000000200040000000100FDFFFEFFFEFFFEFFFDFFFFFF0300FBFF05000200FFFF040002000100FDFF0100FFFF0000FEFFFAFF0700FFFFFDFFFEFF05000000FAFF0500FFFF00000000FCFFFBFF00000000FFFFFDFF0500FAFFFBFF0000000002000000FFFF000003000000FDFF040000000200FDFFFEFF040004000000FCFF0000FDFFFCFFFAFF0000000000000000000000000100FEFF00000100FAFF0000FEFFFCFFFCFF0100030001000300FDFF020000000000FBFF00000200"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-2, -3], [5, 4]], [[0, 0], [-3, 0]], [[-1, 0], [0, 1]], [[-4, 1], [-5, 1]], [[-3, 0], [2, 1]]]> : tensor<5x2x2xi16>}> : () -> tensor<5x2x2xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0500FEFFFEFFFDFF0200000002000000FEFF04000200030002000200FFFF04000000FDFF0100FAFFFFFF000000000100040003000100FEFF0500FFFF0400FEFF0000FFFFFFFF0000FDFF010002000100FDFFFCFF000000000200FDFFFDFF0000FFFFFCFF01000000F7FF010002000000FAFFFFFFFAFF000001000300FBFFFDFF0000FDFF0200FAFF02000100FDFF000000000100FBFF06000300000000000000FCFF0300FCFF02000200FFFF0000FDFF05000200030003000100010000000200000004000100FCFFFEFF0000030003000000000000000000FFFF0200000003000000FEFF00000100010000000200040000000100FDFFFEFFFEFFFEFFFDFFFCFF0300FBFF05000200FFFF040002000100FDFF0100FFFF0000FEFFFAFF07000100FDFFFEFF05000000FAFF0500FFFF00000000FCFFFBFF00000000FFFFFDFF0500FAFFFBFF0000000002000000FFFF00000300FDFFFDFF040000000200FDFFFEFF040001000000FCFF0000FDFFFCFFFAFF0000000000000000000000000100FEFF00000100FAFF00000200FCFFFCFF0100030001000300FDFF020000000000FBFF00000200"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

