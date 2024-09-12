"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi16>, tensor<1x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFAFFFDFF0100020000000000000001000000040002000100FBFF0000040002000300FFFFFCFF030000000200000000000100FFFF0000030006000100FDFF0000010001000300FCFFFEFFFEFF030004000000FFFF01000000FEFFFEFF0000FEFFFBFF0200FAFF0000FFFF00000200FCFF00000000FEFF0000000001000200FEFFFEFFFCFF00000000FFFFFEFF00000100000000000200000002000400FFFFFFFF00000200020002000100060001000400FDFF0100FFFF02000300000000000000FEFFFCFF00000100FFFFFDFFFFFF080003000100FEFF050000000000000001000400000003000100030000000000FFFFFCFF00000200FDFF0000FAFF0100FFFF0000FDFF00000000040003000000030004000000040002000200FDFF0200020002000000000000000000FCFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 4, 4]]> : tensor<1x3xi16>}> : () -> tensor<1x3xi16>
    "func.return"(%1, %2) : (tensor<1x50x3xi16>, tensor<1x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFAFFFDFF0100020000000000000001000000040002000100FBFF0000040002000300FFFFFCFF030000000200000000000100FFFF0000030006000100FDFF0000010001000300FCFFFEFFFEFF030004000000FFFF01000000FEFFFEFF0000FEFFFBFF0200FAFF0000FFFF00000200FCFF00000000FEFF0000000001000200FEFFFEFFFCFF00000000FFFFFEFF00000100000000000200000002000400FFFFFFFF00000200020002000100060001000400FDFF0100FFFF020003000000000000000000F0FF00000100FFFFFDFFFFFF080003000100FEFF050000000000000001000400000003000100030000000000FFFFFCFF00000200FDFF0000FAFF0100FFFF0000FDFF00000000040003000000030004000000040002000200FDFF0200020002000000000000000000FCFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    "func.return"(%0) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

