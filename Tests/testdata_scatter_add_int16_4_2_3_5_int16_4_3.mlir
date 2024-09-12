"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFEFF00000100FDFF0000FBFFFFFF0000040000000100000004000000FFFFFDFF0100FDFFFFFF0000FFFF00000200FFFF0300010000000000FDFF0300FEFF03000400FEFF0100000002000000FEFF0100FEFF0300FFFFFCFFFEFFFEFFFCFF0500FEFFFEFF03000300000002000100FFFF0000010000000000FDFF0000040004000400FEFF0200FDFFFFFF0500FEFF0400030001000000FFFFFFFF040002000000FFFF00000000FFFF02000100000005000000FCFF000001000100FDFF000000000000FDFF010000000000FEFF0700FFFF050001000000040000000500FBFF0000FFFF050002000100FBFFFFFF04000100"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2, 3, 3], [-1, -1, -1], [0, -2, -1], [-2, 3, -2]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi16>, tensor<4x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFEFF00000100FDFFFEFFFBFFFFFF00000400030001000000040000000200FDFF0100FDFFFFFF0000FFFF00000200FFFF0300010000000000FDFF0300FEFF03000400FEFF0000000002000000FEFF0000FEFF0300FFFFFCFFFDFFFEFFFCFF0500FEFFFEFF03000300000002000100FFFF0000010000000000FDFF0000040004000400FEFF0200FDFFFFFF0300FEFF040003000100FFFFFFFFFFFF040002000000FFFF00000000FFFF02000100000005000000FCFF000001000100FDFFFEFF00000000FDFF010003000000FEFF0700FFFF030001000000040000000500FBFF0000FFFF050002000100FBFFFFFF04000100"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    "func.return"(%0) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

