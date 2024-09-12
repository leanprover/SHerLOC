"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0000FAFF000003000000000003000600FFFF01000100FCFF02000000050002000000FFFF000000000100000000000100FFFFFBFF040000000500FDFF0000FDFF0100FEFF0200FEFFFDFFFDFF0000FFFFFFFF0000FDFFFCFF00000000FCFF0300FFFFFEFF0000030000000000FDFF0500FBFF0000020000000500FEFF01000000030000000000FFFFFDFFFDFF0100FBFF01000000FBFFFCFFFEFFFFFFFBFF0500FCFFFEFFFDFFFEFF0000FCFF040002000000050000000000FFFFFFFF0200FEFF01000300FFFFFAFFFFFF01000000FFFF00000000FCFF00000000FEFFFBFFFEFF0200FFFFFEFFFBFF0300000004000200"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, -1, -1], [4, 2, -1], [3, 0, 2], [0, 0, -1]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi16>, tensor<4x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0000FAFF00000300FFFF000003000600FFFFFFFF0100FCFF02000000FFFF02000000FFFF000000000100000000000100FFFFFBFF040000000500FDFF0000FDFF0100FEFF0200FEFFFDFFFDFF0000FFFFFFFF0000FDFFFCFFFFFF0000FCFF0300FFFFFEFF0000030000000000FDFF0500FBFF0000020000000500FEFF01000000030000000000FFFFFDFFFDFF0100FBFF01000000FBFFFCFFFEFFFFFFFBFF0500FCFFFEFFFDFFFEFF0000FCFF040002000000050000000000FFFFFFFF0000FEFF01000300FFFFFAFFFFFF01000000FFFFFFFF0000FCFF00000000FEFFFBFFFEFF0200FFFFFEFFFBFF0300000004000200"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    "func.return"(%0) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

