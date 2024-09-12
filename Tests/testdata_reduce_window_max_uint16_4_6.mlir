"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xui16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xui16>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%6) : (tensor<ui16>) -> ()
    }) : (tensor<4x6xui16>, tensor<ui16>) -> tensor<3x5xui16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xui16>, tensor<3x5xui16>) -> ()
    "func.return"(%5) : (tensor<3x5xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3, 0, 0, 0, 4, 0], [0, 4, 5, 2, 3, 1], [2, 4, 3, 3, 0, 2], [4, 3, 0, 0, 3, 0]]> : tensor<4x6xui16>}> : () -> tensor<4x6xui16>
    "func.return"(%1) : (tensor<4x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[4, 5, 5, 4, 4], [4, 5, 5, 3, 3], [4, 4, 3, 3, 3]]> : tensor<3x5xui16>}> : () -> tensor<3x5xui16>
    "func.return"(%0) : (tensor<3x5xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

