"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    "func.return"(%6) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi16>, tensor<1xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFF00000000FFFF0000FFFF00000000FFFF0000FDFF0700FAFFF9FF0300FBFFFEFF000001000100FCFFFEFFFAFF01000000FDFF05000200020000000000FAFF0000FFFF0500FEFF0000020000000300FAFF0000020000000000FFFFFFFFFEFF040001000000FFFF04000000FBFF0200FEFF0000030002000000030005000000FFFF0000000002000000FDFFFEFFFDFF02000300FCFF030005000200000001000000000002000000FFFFFCFFFCFF02000400FEFF0200FDFF01000200060002000100FFFF0000FCFFFFFF0000FEFF0600FFFF0000FEFF0000FDFF0000000000000400FDFFFEFFFEFF0000FEFFFEFF0000FCFF0000010000000500"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    %2 = "stablehlo.constant"() <{value = dense<-1> : tensor<1xi16>}> : () -> tensor<1xi16>
    "func.return"(%1, %2) : (tensor<1x125xi16>, tensor<1xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010000000000FFFF0000FFFF00000000FFFF0000FDFF0700FAFFF9FF0300FBFFFEFF000001000100FCFFFEFFFAFF01000000FDFF05000200020000000000FAFF0000FFFF0500FEFF0000020000000300FAFF0000020000000000FFFFFFFFFEFF040001000000FFFF04000000FBFF0200FEFF0000030002000000030005000000FFFF0000000002000000FDFFFEFFFDFF02000300FCFF030005000200000001000000000002000000FFFFFCFFFCFF02000400FEFF0200FDFF01000200060002000100FFFF0000FCFFFFFF0000FEFF0600FFFF0000FEFF0000FDFF0000000000000400FDFFFEFFFEFF0000FEFFFEFF0000FCFF0000010000000500"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    "func.return"(%0) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

