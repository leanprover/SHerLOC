"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      "stablehlo.return"(%arg1) : (tensor<i16>) -> ()
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFFFCFF0000FDFF0000FFFF0000010002000000000005000600FDFFFFFFFFFFFEFFFEFF0300FEFF010000000000FFFF00000100020002000000010000000000000000000000FCFFFAFF00000500000000000000FFFF000000000400010004000500FDFF040000000400FDFF03000100FFFF040000000400FEFFFEFF0200060000000000FBFFFFFF00000100010000000300FDFF0200000005000200FFFF0200FCFF030001000300F9FF0000000002000700FFFFFEFF0300FFFFFAFF010000000000000002000000FCFFFFFF0200FEFFFEFF0200FCFF01000000FFFF01000000000006000000FBFFFBFF0000FEFFFFFF"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2, 0, 6], [-3, 0, -3], [-2, 2, 2], [-1, -1, 0]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi16>, tensor<4x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFFFFCFF0000FDFFFEFFFFFF0000010002000000000005000600FDFF0600FFFFFEFFFEFF0300FEFF010000000000FFFF0000010002000200000001000000000000000000FDFFFCFFFAFF00000500000000000000FFFF0000FDFF0400010004000500FDFF040000000400FDFF03000100FFFF040000000400FEFFFEFF02000600FEFF0000FBFFFFFF00000200010000000300FDFF0200000005000200FFFF0200FCFF030001000300F9FF0000000002000700FFFFFEFF0300FFFFFAFFFFFF0000000000000200FFFFFCFFFFFF0200FEFF00000200FCFF01000000FFFF01000000000006000000FBFFFBFF0000FEFFFFFF"> : tensor<4x2x3x5xi16>}> : () -> tensor<4x2x3x5xi16>
    "func.return"(%0) : (tensor<4x2x3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

