"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFFFBFF0000FCFF010001000300FFFFFCFFF9FFFFFFFCFF00000000000004000400FFFFFCFF03000000FFFF000000000100FDFF0300FEFFFFFFFBFFFFFFFFFF05000200FDFFFCFFFFFFFAFF0000FEFF0800000002000300FEFFFBFF0300FDFF0000020000000200000002000100FFFFFCFFFEFF03000100FFFF000001000500FDFF0000FFFF0300FDFF05000000FEFFFEFF0100020000000100FDFF0300030001000000FFFFFDFF0100FDFF0300F9FFFFFF0400030003000700FEFFFEFF010001000400010000000100FFFFFDFF0100000000000000F7FF000005000300000003000200FAFFFAFFFEFF00000000FBFF"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-6, 1, 0], [1, 8, 0], [5, 1, 3], [0, -5, 0]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi16>, tensor<4x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFFFFBFF0000FCFF010001000300FFFFFCFF0100FFFFFCFF00000000000004000400FFFFFCFF03000000FFFF000000000100FDFF0300FEFFFFFFFBFFFFFFFFFF050002000100FCFFFFFFFAFF0000080008000000020003000000FBFF0300FDFF0000020000000200000002000100FFFFFCFFFEFF03000100FFFF00000100050005000000FFFF0300FDFF05000000FEFFFEFF0100030000000100FDFF0300030001000000FFFFFDFF0100FDFF0300F9FFFFFF0400030003000700FEFF0000010001000400010000000100FFFFFDFF0100000000000000F7FF000005000300000003000200FAFFFAFFFEFF00000000FBFF"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    "func.return"(%0) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

