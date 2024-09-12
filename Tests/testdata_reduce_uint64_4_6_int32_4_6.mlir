"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6xui64>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xui64>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6xui64>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<ui64>}> : () -> tensor<ui64>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%7) : (tensor<ui64>) -> ()
    }) : (tensor<4x6xui64>, tensor<ui64>) -> tensor<6xui64>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xui64>, tensor<6xui64>) -> ()
    "func.return"(%6) : (tensor<6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 4, 1, 1, 4, 2], [3, 1, 2, 1, 1, 5], [1, 1, 1, 3, 0, 1], [5, 5, 1, 3, 1, 1]]> : tensor<4x6xui64>}> : () -> tensor<4x6xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[-4, -2, 2, 1, 0, 5], [0, 0, 0, 0, 0, -2], [-4, 3, 0, -6, 4, 1], [-1, 0, -2, -2, -4, 0]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[12, 14, 8, 11, 9, 12]> : tensor<6xui64>}> : () -> tensor<6xui64>
    "func.return"(%0) : (tensor<6xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

