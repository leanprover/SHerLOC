"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2x2xi64>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000002FFFE03FC03FF02FD03000002FE0302FEFFFC00000000FE0101FCFFFC0402000101020104FCFD00FF03FF000103020003FF06000000FFFF0100FA02FE02FDFE04000000060000FBFD0203FEFFFCFF06FBFB01040005000000000300000504FFFDFF0303FF0600FD01000000000102FAFEFE0002FD04FFFB00FA01FF02FDFE0400FF0103FEFF020103FBFFFBFFFCFB000000FE040100FDFEFAFC02FD000507FE000002FC030400FCFCFF0000FC0000F9FEFCF9030105040000FEFE000400FCFF00FDFE0705FD04FDFE000306FE00FD00"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[[2, -1], [2, 1]], [[-1, 3], [1, 1]], [[4, -4], [0, 1]], [[0, -5], [0, 0]], [[-4, 1], [-2, -6]]]> : tensor<5x2x2xi8>}> : () -> tensor<5x2x2xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<5x2x2xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000002FFFE03FC03FF01FD03000002FE03FFFEFFFC00000000FE0101FCFFFC0402000101020104FCFD00FFFFFF000103020003FF06000000FFFF0100FA02FE02FDFE04000000010000FBFD0203FEFFFCFF06FBFB01040005000000000300000504FFFDFF03FCFF0600FD01000000000100FAFEFE0002FD04FFFB00FA01FF02FDFE0400FF0103FEFF020103FBFFFBFFFBFB000000FE040100FDFEFAFC02FD000507FE000002FC030400FCFCFF0000FC0000F9FEFCF9030105040000FEFE000400FCFF00FDFE0705FD04FDFE000306FE00FD00"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

