"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000502FFFE0102FF00020203010200020002FF00FE03FFFE0002FF0202FE000400FEFFFDFAFD00FCFFFA00FE03FF03FFFFFE02FD00000000040300FE00010101FE070101FBFD0200FE000300FC0203FD0403FEFF00FF0004FE0101FF00000106FFFD000100FBFFFF03FF01FF04FE08FA0000020300FB0400FFFFFE00010402FA0200FF00FE010000000100040406FFFFFD0000FD00FEFF0503FFFB0400FB0100FE01FF0103050000FF0001FF00FC020300FB0300FE01FEFB0003FEFFFB030000010100FFFDFF0304FE00FEFE02FF00FF0000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, -1, -1, 1, 2, 0, 0], [0, 1, 0, 4, 2, 0, 1]]> : tensor<2x7xi8>}> : () -> tensor<2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000502FFFE01020000FE0206000000020002FF00FE03FFFE0002FF0202FE000400FEFFFDFAFD00FCFFFA00FE03FF03FFFFFE02FD00000000040300FE00010101FE070101FBFD0200FE000300FC0203FD0403FEFF00FF0004FE0101FF00000106FFFD000100FBFFFF0300010010FC00FA0000020300FB0400FFFFFE00010402FA0200FF00FE010000000100040406FFFFFD0000FD00FEFF0503FFFB0400FB0100FE01FF0103050000FF0001FF00FC020300FB0300FE01FEFB0003FEFFFB030000010100FFFDFF0304FE00FEFE02FF00FF0000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

