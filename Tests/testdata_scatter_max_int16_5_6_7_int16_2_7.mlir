"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2xi64>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x04000200FFFFFCFF0000FEFFFDFFFFFFF8FF02000100FDFFFFFFF8FFFFFF010001000000FCFF0000000001000200000000000100FBFF01000200FFFFFAFFFCFFFBFF0000FFFF010000000000000002000300FDFFFFFF03000400FDFF0000FDFFFDFF04000100FEFF000004000200000000000000FFFF000003000000010000000300FDFF0700FAFF0000FFFFFFFF0000FEFFFBFF0500FFFFFAFF0000000001000000FEFF0100FAFF05000100020002000100FEFFFFFF000005000200FFFF0300000003000000010003000000FDFF0300FEFFFFFF0200FBFFFBFF03000100FEFFFCFFFEFF0000FEFF03000000FDFF00000400FFFFFFFF01000200000000000200010000000000FEFF040002000000000000000000020004000300020001000100FEFF0000FFFF01000100000000000200000000000000050001000000FFFF01000200FCFF040000000400FDFF0000FDFF00000200010000000000030000000300FDFF040000000200FEFF0800FCFF01000000FEFFFEFFFAFF0000000001000200FFFF0100FEFFFEFFFEFFF9FF0200FEFF04000000FEFFFEFF000003000500FFFFFAFF0400"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 0, 1, -2, 1, 5], [0, -3, 2, -3, -2, 1, 0]]> : tensor<2x7xi16>}> : () -> tensor<2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x04000200FFFFFCFF0000FEFFFDFF0000010002000100FEFF01000500FFFF010001000000FCFF0000000001000200000000000100FBFF01000200FFFFFAFFFCFFFBFF0000FFFF010000000000000002000300FDFFFFFF03000400FDFF0000FDFFFDFF04000100FEFF000004000200000000000000FFFF000003000000010000000300FDFF0700FAFF0000FFFFFFFF0000FEFFFBFF0500FFFFFAFF0000000001000000FEFF0100FAFF05000100020002000100FEFFFFFF000005000200FFFF0300000003000000010003000000FDFF0300FEFF000002000200FDFF030001000000FCFFFEFF0000FEFF03000000FDFF00000400FFFFFFFF01000200000000000200010000000000FEFF040002000000000000000000020004000300020001000100FEFF0000FFFF01000100000000000200000000000000050001000000FFFF01000200FCFF040000000400FDFF0000FDFF00000200010000000000030000000300FDFF040000000200FEFF0800FCFF01000000FEFFFEFFFAFF0000000001000200FFFF0100FEFFFEFFFEFFF9FF0200FEFF04000000FEFFFEFF000003000500FFFFFAFF0400"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

