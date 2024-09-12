"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    "func.return"(%6) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi16>, tensor<1xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0100FCFF000002000100010002000300010000000200FFFFFFFF0300FCFF0400010000000400FFFF0200FBFF0500FEFF0100FEFF030000000200FEFF00000200FCFFFFFF0400FCFF0300020002000400FFFFFDFFFFFF0100020000000400FFFF0100FEFF02000100FEFF000001000000FBFF04000100FCFFFCFF00000300FFFF05000400FFFF04000000FDFF0300FEFF0500FEFF01000300FEFF030001000000FEFF00000200FFFF0000FFFF0100FFFFFFFF0300FFFF00000000000003000000020001000500FEFF02000000FAFF02000100010001000200FDFF03000400FCFF00000300000005000000FDFF0000010000000100040000000000"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    "func.return"(%1, %2) : (tensor<1x125xi16>, tensor<1xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0100FCFF000002000100010002000300010000000200FFFFFFFF0300FCFF0400010000000400FFFF0200FBFF0500FEFF0100FEFF030000000200FEFF00000200FCFFFFFF0400FCFF0300020002000400FFFFFDFFFFFF0100020000000400FFFF0100FEFF02000100FEFF000001000000FBFF04000100FCFFFCFF00000300FFFF05000400FFFF04000000FDFF0300FEFF0500FEFF01000300FEFF030001000000FEFF00000200FFFF0000FFFF0100FFFFFFFF0300FFFF00000000000003000000020001000500FEFF02000000FAFF02000100010001000200FDFF03000400FCFF00000300000005000000FDFF0000010000000100040000000000"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    "func.return"(%0) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

