"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    "func.return"(%6) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi16>, tensor<1xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0500FFFF02000300FDFF01000100FDFFFFFF00000000FFFF040000000000FEFF000001000000FBFF0000010000000000FEFFFFFF070002000400030000000000FAFF0000FEFF0100FBFF00000400FEFFFFFFFDFF00000000FDFFFFFFFCFF0000FCFF0200FDFF0200FFFF0400010004000000010000000000F8FFFFFFFFFF00000000FEFF04000000FDFFFEFFFEFF00000000000002000000FEFF000003000100FEFFFEFF0000000001000000FEFFFFFF0000FEFFFAFFFDFF02000200FFFFFEFFFBFFFFFFFEFFFEFF02000200FFFFFBFF0100FFFF0300FFFF010004000300FFFF00000000040002000000FFFF020006000900FEFF0200FBFF0500"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    %2 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi16>}> : () -> tensor<1xi16>
    "func.return"(%1, %2) : (tensor<1x125xi16>, tensor<1xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0300FFFF02000300FDFF01000100FDFFFFFF00000000FFFF040000000000FEFF000001000000FBFF0000010000000000FEFFFFFF070002000400030000000000FAFF0000FEFF0100FBFF00000400FEFFFFFFFDFF00000000FDFFFFFFFCFF0000FCFF0200FDFF0200FFFF0400010004000000010000000000F8FFFFFFFFFF00000000FEFF04000000FDFFFEFFFEFF00000000000002000000FEFF000003000100FEFFFEFF0000000001000000FEFFFFFF0000FEFFFAFFFDFF02000200FFFFFEFFFBFFFFFFFEFFFEFF02000200FFFFFBFF0100FFFF0300FFFF010004000300FFFF00000000040002000000FFFF020006000900FEFF0200FBFF0500"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    "func.return"(%0) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

