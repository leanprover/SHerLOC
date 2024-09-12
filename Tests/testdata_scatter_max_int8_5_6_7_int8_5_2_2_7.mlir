"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02FD01FFFFFF04FF03FFFF000403FEFEFD0001FFFE02FD00FCFB04FF010200FD0000FDFFFE01FE010502FE01FD00FF01FE04000007020100000100030005FE000301010300000000FF0002FD00FE0105FB000300FC0000040400000004FFFB0000FE0001010403FD00FE0205FFFDFC00020000FC000003FEFC00FE01010100FD0004FF02FD0100FFFA0200000403FCFB00050204FC01040200FEFF0202FEFDFF03FB0300FD00FEFEFC0001FEFC01000000FE00FD05FDFD0000FC00000002FDFFFD02FE04FE0506FF0100FFFCFEFE01000002"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<"0xFEFE02FB000001FC02FEFCFB0201FF02FDFF03FE0102030101FCFD0001000103FF00FBFDFE000103FD02FE00FFFD020300FE010306000103000004000000050002FDFE01FF06FFFE0000FDFD000004000600FFFFFD00FDFE05FE030200FE0201010101FCFC01FBFE010001FDFB0004FDFEFF01FE010102FEFD0400020100FBFB01FFFF00FD010005FEFCFDFF"> : tensor<5x2x2x7xi8>}> : () -> tensor<5x2x2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02FE02FF000004FF03FFFF000403FF02FD0003FF0102030101FC0400010200FD0000FDFFFE01FE01050201010103FF01FE0400000703010200010003020500000303060301030000FF0002FD00FE0105FB000300000004040400050004FFFE0100060001010403FD000004050600FF00020000FC000003FEFC00FE010101000000040502030200FF020201010403FC01000502040101040204FEFF0202FEFDFF03FB0300FD00FEFEFE0001FE01010200000400020500FD0001FF000000020005FE02FE04FE0506FF0100FFFCFEFE01000002"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

