"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02000000FFFFFFFF0000020004000300000001000000FEFF02000100000000000000020003000100F7FFFCFF0000FCFFFFFF020000000300FEFFFFFF0300FAFF020003000500FFFFFAFF00000300FFFF0000FFFFFCFFFAFFFAFFFAFFFAFFFFFF010001000500FEFF0000FDFFFFFF00000100FDFFFCFF03000000FEFF0000F8FFFDFF00000000FCFF00000000FEFF0000F7FFFEFF0200000000000100FDFFFEFF0000FEFFFCFF010003000100FEFF020001000100FDFF000005000000000000000000FFFFFEFFFBFF0500FFFFFFFF010003000000FBFF00000200FEFF0000FDFFFFFFFFFF0100FCFF00000100FDFF0100"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[6, 2, -2], [-1, -2, -3], [2, 0, 4], [-4, 0, 2]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi16>, tensor<4x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02000000FFFFFFFF0000020004000300000002000000FEFF02000100000000000000020003000100F7FFFCFF0000FCFFFFFF020000000300FEFFFFFF0300FAFF02000300FBFFFFFFFAFF0000030002000000FFFFFCFFFAFF1200FAFFFAFFFFFF010001000500FEFF0000FDFFFFFF00000100FDFFFCFF03000000FEFF0000F8FFFAFF00000000FCFF00000000FEFF0000F7FFFEFF0800000000000100FDFFFEFF0000FEFFFCFF010003000100FEFF020001000100FDFF000005000000000000000000FFFFFEFF00000500FFFFFFFF010006000000FBFF00000200FEFF0000FDFFFFFFFFFF0100FCFF00000100FDFF0100"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    "func.return"(%0) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

