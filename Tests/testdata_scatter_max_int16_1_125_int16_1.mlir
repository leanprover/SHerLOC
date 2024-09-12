"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    "func.return"(%6) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi16>, tensor<1xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x030000000000FAFFFDFF00000000000005000000FDFF0300000000000200060004000000FCFFFEFFFFFFFFFFFFFF0100030000000100FEFF02000100FEFF00000000FFFFFDFFFEFF01000000FFFF000002000000FEFF0200000004000100FFFFFFFF0200FDFFFEFF040003000200010003000500FFFF000000000000FFFF020003000000FEFF0100FDFF0000FFFFFBFF03000700FFFF000005000100FDFF01000100FBFF00000100000002000200FDFF040000000600FEFF01000000020007000000FFFF01000000020000000000020002000400FEFF06000300050000000000FEFF01000100FCFF000001000000000003000000FDFF00000100"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    %2 = "stablehlo.constant"() <{value = dense<-2> : tensor<1xi16>}> : () -> tensor<1xi16>
    "func.return"(%1, %2) : (tensor<1x125xi16>, tensor<1xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x030000000000FAFFFDFF00000000000005000000FDFF0300000000000200060004000000FCFFFEFFFFFFFFFFFFFF0100030000000100FEFF02000100FEFF00000000FFFFFDFFFEFF01000000FFFF000002000000FEFF0200000004000100FFFFFFFF0200FDFFFEFF040003000200010003000500FFFF000000000000FFFF020003000000FEFF0100FDFF0000FFFFFBFF03000700FFFF000005000100FDFF01000100FBFF00000100000002000200FDFF040000000600FEFF01000000020007000000FFFF01000000020000000000020002000400FEFF06000300050000000000FEFF01000100FCFF000001000000000003000000FDFF00000100"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    "func.return"(%0) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

