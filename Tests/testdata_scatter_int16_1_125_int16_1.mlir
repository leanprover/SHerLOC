"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      "stablehlo.return"(%arg1) : (tensor<i16>) -> ()
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    "func.return"(%6) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi16>, tensor<1xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x00000000FEFFFCFF000007000100FBFF02000400FDFFFFFF010000000000FFFF0500000002000300060003000300FFFFFFFFFFFF000003000200020004000000FFFFFFFF0000000000000200FCFF04000100000006000000FFFF02000300FFFF0000050002000000FEFF02000000020004000100FEFFFBFF03000500FEFFFEFF0000FEFF00000100000000000100FFFFFDFFFEFF010002000400FEFF03000500FFFFFFFF030004000200FFFF0200FBFFFFFF000006000500FFFF0000FBFF01000400FDFF000000000400FEFFFFFF01000400FCFF0000FAFF0300FDFF0100020006000000FDFFFFFFFEFF01000200FDFFFFFF0200FCFFFDFF0000"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    "func.return"(%1, %2) : (tensor<1x125xi16>, tensor<1xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00000000FEFFFCFF000007000100FBFF02000400FDFFFFFF010000000000FFFF0500000002000300060003000300FFFFFFFFFFFF000003000200020004000000FFFFFFFF0000000000000200FCFF04000100000006000000FFFF02000300FFFF0000050002000000FEFF02000000020004000100FEFFFBFF03000500FEFFFEFF0000FEFF00000100000000000100FFFFFDFFFEFF010002000400FEFF03000500FFFFFFFF030004000200FFFF0200FBFFFFFF000006000500FFFF0000FBFF01000400FDFF000000000400FEFFFFFF01000400FCFF0000FAFF0300FDFF0100020006000000FDFFFFFFFEFF01000200FDFFFFFF0200FCFFFDFF0000"> : tensor<1x125xi16>}> : () -> tensor<1x125xi16>
    "func.return"(%0) : (tensor<1x125xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

