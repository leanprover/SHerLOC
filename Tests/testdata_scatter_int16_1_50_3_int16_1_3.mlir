"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      "stablehlo.return"(%arg1) : (tensor<i16>) -> ()
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi16>, tensor<1x3xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x06000000040000000100FEFF0000FEFF020004000000FFFFFFFF0500020005000000FFFF05000000FEFFF7FF0000000000000200FBFFFEFFFFFF06000000FFFF02000100FEFF01000700FEFFFCFF0400FEFF02000000FEFF0000FDFFFCFFFCFF01000000FFFFFCFF0200FFFF0000050000000000FFFF06000400FFFF0000030004000100F9FFFAFF00000200010001000200FEFFFDFF0200F9FF0200FCFF00000300FFFFFEFFFDFFFEFF050004000200FFFF01000500000002000100FEFFFEFFFDFF00000000FEFF04000100050003000200FEFFFEFF0600FCFFFFFF0000010005000200000001000100FEFFFEFF0200FDFFFCFF0000FDFFFCFF0500F9FF020007000400050003000000FEFF00000300FDFFFBFF00000100FFFFFFFFFDFFFEFF000000000200FFFF0200FEFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-6, 1, 0]]> : tensor<1x3xi16>}> : () -> tensor<1x3xi16>
    "func.return"(%1, %2) : (tensor<1x50x3xi16>, tensor<1x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x06000000040000000100FEFF0000FEFF020004000000FFFFFFFF0500020005000000FFFF05000000FEFFF7FF0000000000000200FBFFFEFFFFFF06000000FFFF02000100FEFF01000700FEFFFCFF0400FEFF02000000FEFF0000FDFFFCFFFCFF01000000FFFFFCFF0200FFFF0000050000000000FFFF06000400FFFF0000030004000100F9FFFAFF00000200010001000200FEFFFDFF0200F9FF0200FCFF00000300FFFFFEFFFDFFFEFF050004000200FFFF01000500000002000100FEFFFEFFFAFF01000000FEFF04000100050003000200FEFFFEFF0600FCFFFFFF0000010005000200000001000100FEFFFEFF0200FDFFFCFF0000FDFFFCFF0500F9FF020007000400050003000000FEFF00000300FDFFFBFF00000100FFFFFFFFFDFFFEFF000000000200FFFF0200FEFF"> : tensor<1x50x3xi16>}> : () -> tensor<1x50x3xi16>
    "func.return"(%0) : (tensor<1x50x3xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

