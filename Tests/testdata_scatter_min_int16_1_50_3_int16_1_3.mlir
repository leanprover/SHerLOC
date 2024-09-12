"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi16>, tensor<1x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFF0000FDFFFEFF0100FFFF00000000FEFF00000000FFFF020000000000FDFF0000FEFF000000000000000002000400FEFF02000200000000000200030000000200FCFF0300FCFF01000200010000000200FEFFFAFF0000FFFFFEFF0000FCFF0000FDFFFEFFFFFF00000000FEFFFDFFFDFF0000FDFFFFFFFFFF0600FCFF0200FEFF0000FEFFFDFFFBFF0200FFFF01000300FCFF0600000000000100FFFF00000000000000000000FFFF0200FDFFFFFF0200F8FF00000000FDFF0000FFFF0300FDFF0000FDFF00000000FFFF0200FCFF020000000300FEFFFEFFFEFFFFFFFEFFFCFFFEFF0400FFFFFEFFFFFFFCFF0000FCFFFEFFFEFF02000100010000000000000001000300FDFF030005000500FDFF02000000FEFF0100FDFF03000500030000000400FEFFFFFFFEFFFEFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[3, -5, 2]]> : tensor<1x3xi16>}> : () -> tensor<1x3xi16>
    "func.return"(%1, %2) : (tensor<1x50x3xi16>, tensor<1x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFFF0000FDFFFEFF0100FFFF00000000FEFF00000000FFFF020000000000FDFF0000FEFF000000000000000002000400FEFF02000200000000000200030000000200FCFF0300FCFF01000200010000000200FEFFFAFF0000FFFFFEFF0000FCFF0000FDFFFEFFFFFF00000000FEFFFDFFFDFF0000FDFFFFFFFFFF0600FCFF0200FEFF0000FEFFFDFFFBFF0200FFFF01000300FCFF0600000000000100FFFF00000000000000000000FFFF0200FDFFFFFF0200F8FF00000000FDFF0000FFFF0300FDFFFBFFFDFF00000000FFFF0200FCFF020000000300FEFFFEFFFEFFFFFFFEFFFCFFFEFF0400FFFFFEFFFFFFFCFF0000FCFFFEFFFEFF02000100010000000000000001000300FDFF030005000500FDFF02000000FEFF0100FDFF03000500030000000400FEFFFFFFFEFFFEFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    "func.return"(%0) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

