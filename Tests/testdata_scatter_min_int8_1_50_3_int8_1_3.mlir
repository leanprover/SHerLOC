"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi8>, tensor<1x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFAFFFE05FBFD0002FD04000200FB00F9FEFEFFFFFBFE000100FD000001FF00FF0001000200000203FCFD00FFFFFF05FD00020100000002FC0004FEFD00FFFD0001030000FE000301FAFEFBFD01FF03FFFF00FD000000FBFDFC0003FFFDFBFE000000FF0005FDFE01FD00FD01FDFF00FD0001FF0000FEFF0001FCFEFEFF03050000000003FF0000FE0000FC03FE010001FCFFFA000100"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-3, 1, 3]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
    "func.return"(%1, %2) : (tensor<1x50x3xi8>, tensor<1x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFAFFFE05FBFD0002FD04000200FB00F9FEFEFFFFFBFE000100FD000001FF00FF0001000200000203FCFD00FFFFFF05FD00020100000002FC0004FEFD00FFFD0001030000FE000301FAFEFBFD01FF03FFFF00FD000000FBFDFC0003FFFDFBFE00FD00FF0005FDFE01FD00FD01FDFF00FD0001FF0000FEFF0001FCFEFEFF03050000000003FF0000FE0000FC03FE010001FCFFFA000100"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    "func.return"(%0) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

