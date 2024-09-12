"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      "stablehlo.return"(%arg1) : (tensor<i8>) -> ()
    }) : (tensor<1x125xi8>, tensor<1xi64>, tensor<1xi8>) -> tensor<1x125xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi8>, tensor<1x125xi8>) -> ()
    "func.return"(%6) : (tensor<1x125xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi8>, tensor<1xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFBFCFFFF01030201000101FE0006FDFF02070002FEFE0300FAFC0602FF0000FF0100FFFB040002FF00FE000404FC0800000200FDFC000000FEFF0105FEFCFE03FC020002FE0500FD050005020403FE0302FD06FE0400020101FE0001FEFF0305FD0000060000FE01FEFFFE000005FC02010102000201010001FF0404FE"> : tensor<1x125xi8>}> : () -> tensor<1x125xi8>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    "func.return"(%1, %2) : (tensor<1x125xi8>, tensor<1xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00FCFFFF01030201000101FE0006FDFF02070002FEFE0300FAFC0602FF0000FF0100FFFB040002FF00FE000404FC0800000200FDFC000000FEFF0105FEFCFE03FC020002FE0500FD050005020403FE0302FD06FE0400020101FE0001FEFF0305FD0000060000FE01FEFFFE000005FC02010102000201010001FF0404FE"> : tensor<1x125xi8>}> : () -> tensor<1x125xi8>
    "func.return"(%0) : (tensor<1x125xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

