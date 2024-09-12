"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x2xi64>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFDFF00000000020004000300FEFF010000000200FEFF0000FEFF020001000000000000000000FEFF000000000400FDFFFEFF00000300000000000100FEFF0000FBFF000000000100FFFF0500FEFFFDFFFDFF0200FCFF0400FFFF050001000000FDFF07000000FFFFFDFFFCFFFDFFFEFFFFFFFCFFFFFF00000000FEFFF9FF0500020001000000FEFF00000100FEFF00000000000003000200FEFF0300FEFF0000F9FF00000100FFFFFEFF03000600000004000500FFFF01000100050000000000000002000000000002000600050000000300FEFF020000000200FFFFFDFFFEFFFFFF010000000000FEFFFFFF050000000100010003000000FFFFFEFFF9FF0000FBFF00000000020005000000FCFFFFFF0000FDFFFFFF01000000030000000000010003000000FBFFFEFFFFFF0300FFFF05000000020000000000FDFFFFFF040007000000020002000200030000000100FFFFFCFFFFFF000002000100FCFFFEFF00000400000000000000FEFF01000100F8FFFEFFFFFF05000000FDFFFFFF040000000000FDFF0100FDFF010000000000000000000000010000000700FBFFFBFFFFFF0400"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 0], [1, 1]], [[2, -4], [-1, 5]], [[0, 3], [-4, 4]], [[0, -1], [-2, -6]], [[0, -1], [0, -7]]]> : tensor<5x2x2xi16>}> : () -> tensor<5x2x2xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFDFF00000000020004000300FEFF010000000300FEFF0000FEFF020001000000000000000000FEFF000000000400FDFFFEFF00000300000001000100FEFF0000FBFF000000000100FFFF0500FEFFFDFFFDFF0200FCFF0600FFFF050001000000FDFF070000000400FDFFFCFFFDFFFEFFFFFFFCFFFFFFFCFF0000FEFFF9FF0500020001000000FEFF00000100FDFF00000000000003000200FEFF0300FEFF0000F9FF00000100FFFFFEFF03000600000004000500FFFF01000100090000000000000002000000000002000900050000000300FEFF020000000200FFFFFDFFFEFFFBFF010000000000FEFFFFFF050000000100010003000000FFFFFEFFF9FF0000FBFF00000000020005000000FCFFF9FF0000FDFFFFFF0100000003000000FFFF010003000000FBFFFEFFFFFF0300FFFF05000000000000000000FDFFFFFF040007000000020002000200030000000100FFFFFCFFFFFF000002000100FCFFFEFF0000FDFF000000000000FEFF01000100F8FFFDFFFFFF05000000FDFFFFFF040000000000FDFF0100FDFF010000000000000000000000010000000700FBFFFBFFFFFF0400"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

