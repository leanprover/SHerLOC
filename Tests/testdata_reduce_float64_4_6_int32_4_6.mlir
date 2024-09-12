"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xf64>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf64>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xf64>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<3.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<i32>, %arg2: tensor<f64>, %arg3: tensor<i32>):
      %9 = "stablehlo.maximum"(%arg0, %arg2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %10 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%9, %10) : (tensor<f64>, tensor<i32>) -> ()
    }) : (tensor<4x6xf64>, tensor<4x6xi32>, tensor<f64>, tensor<i32>) -> (tensor<6xf64>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6xf64>, tensor<6xf64>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xf64>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf64>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[3.7197398050369812, 0.64320452105575032, 2.2631961699769709, -2.328491537315077, -1.0646327236966586, 0.9983394391370205], [-6.7277944839589292, -2.9801497201356963, 1.1412801151040586, 5.8399824009095695, -1.2944630749197987, 4.9393520979193211], [1.1556124464376827, -3.6421368720551914, 3.6417975059106329, -3.0109402861875258, 0.65566837496731945, 4.0286355796821383], [-5.0392029602908295, 0.85577383132506623, -0.25675898760833782, 2.635548767603793, -4.7856806886501522, -0.28606486625921967]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    %3 = "stablehlo.constant"() <{value = dense<[[3, 0, -1, 2, 0, 1], [8, -7, -4, -2, 0, 3], [-2, 0, 1, 3, 0, 1], [0, -4, -2, -5, -3, 2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xf64>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xf64>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[3.7197398050369812, 3.000000e+00, 3.6417975059106329, 5.8399824009095695, 3.000000e+00, 4.9393520979193211]> : tensor<6xf64>}> : () -> tensor<6xf64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -7, -4, -5, -3, 0]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xf64>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

