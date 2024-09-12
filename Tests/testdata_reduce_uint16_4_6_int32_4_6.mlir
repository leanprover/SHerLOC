"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6xui16>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xui16>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6xui16>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<ui16>}> : () -> tensor<ui16>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<4x6xui16>, tensor<ui16>) -> tensor<6xui16>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xui16>, tensor<6xui16>) -> ()
    "func.return"(%6) : (tensor<6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 5, 4, 0, 1, 5], [7, 0, 1, 0, 1, 2], [3, 1, 0, 4, 3, 1], [2, 3, 0, 3, 5, 0]]> : tensor<4x6xui16>}> : () -> tensor<4x6xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, 6, 3, 0, -2, 2], [-1, -6, -3, 3, 0, -4], [-2, 5, -1, 3, -3, 4], [0, -4, -1, 4, 0, 0]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[17, 12, 8, 10, 13, 11]> : tensor<6xui16>}> : () -> tensor<6xui16>
    "func.return"(%0) : (tensor<6xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

