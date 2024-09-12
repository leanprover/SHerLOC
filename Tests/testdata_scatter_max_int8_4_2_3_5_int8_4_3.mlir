"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFDFE00FE0100000302FF07FDFCFD0000020003FFFE00000202010005FE07FB02000001FF0005FDF9FC0302000301FEFFFE01020103060000FDFFFDFF000100010404FFFF0302FE0202030000FD0000FF01FB030000020104FF00FE0300FF000501FF0000FB010202FDFB020002000000FF00FF000000FC01"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 0], [-4, 0, -2], [4, 0, 0], [0, 1, -2]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi8>, tensor<4x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFDFE00FE01000003020007FDFCFD0000020003FFFE00000202010005FE07FB02000001FF0005FD00FC0302000301FEFFFE01020103060000FDFFFDFF000100010404FFFF0302FE0202030000FD0000FF01FB030000020104FF00FE0300FF000501FF0001FB010202FEFB020002000000FF00FF000000FC01"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    "func.return"(%0) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

