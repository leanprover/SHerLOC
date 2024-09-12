"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xui8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xui8>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<ui8>}> : () -> tensor<ui8>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%6) : (tensor<ui8>) -> ()
    }) : (tensor<4x6xui8>, tensor<ui8>) -> tensor<3x5xui8>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xui8>, tensor<3x5xui8>) -> ()
    "func.return"(%5) : (tensor<3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 1, 2, 0, 1, 1], [0, 0, 2, 1, 0, 3], [1, 2, 0, 0, 2, 3], [0, 1, 0, 1, 3, 4]]> : tensor<4x6xui8>}> : () -> tensor<4x6xui8>
    "func.return"(%1) : (tensor<4x6xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[5, 2, 2, 1, 3], [2, 2, 2, 2, 3], [2, 2, 1, 3, 4]]> : tensor<3x5xui8>}> : () -> tensor<3x5xui8>
    "func.return"(%0) : (tensor<3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

