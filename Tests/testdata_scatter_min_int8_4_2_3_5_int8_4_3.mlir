"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0002FD00FAFF00FA030104FEFD01FF04FE000002FF000200FE01FD0200FE02000501020100FFFF03FFFEFFFD00FFFBFEFDFF0001010000FD0000FBFD000002FD0103010004FE00FEFBFDFDFB01FFFF00FD00000100F9FFFE04FD000400010002FDFE01F80000000102F8FFFB02000000FFFAFC000000FD00"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 2, 0], [0, 0, -1], [-1, 3, 0], [-4, 3, 1]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi8>, tensor<4x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0002FD00FAFF00FA030104FEFD01FF04FE000002FF000200FE01FD0200FE02000501000100FFFF00FFFEFFFDFFFFFBFEFDFF0001010000FD0000FBFD000002FDFF03010004FE00FEFBFDFDFB01FFFF00FD00000100F9FFFE04FD00040001FC02FDFE01F80000000101F8FFFB02000000FFFAFC000000FD00"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    "func.return"(%0) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

