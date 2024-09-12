"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<6xf32>, tensor<6xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf32>, tensor<4x6xi32>)
    %5:2 = "func.call"() <{callee = @expected}> : () -> (tensor<6xf32>, tensor<6xi32>)
    %6 = "stablehlo.constant"() <{value = dense<3.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%4#0, %4#1, %6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<i32>):
      %9 = "stablehlo.maximum"(%arg0, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %10 = "stablehlo.minimum"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%9, %10) : (tensor<f32>, tensor<i32>) -> ()
    }) : (tensor<4x6xf32>, tensor<4x6xi32>, tensor<f32>, tensor<i32>) -> (tensor<6xf32>, tensor<6xi32>)
    "stablehlo.custom_call"(%8#0, %5#0) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6xf32>, tensor<6xf32>) -> ()
    "stablehlo.custom_call"(%8#1, %5#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi32>, tensor<6xi32>) -> ()
    "func.return"(%8#0, %8#1) : (tensor<6xf32>, tensor<6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf32>, tensor<4x6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[-0.310264707, -1.99879205, -1.81809437, -0.426923692, -0.390648484, 1.98554444], [-1.17653537, -5.20831203, 0.447554022, -2.48738694, -2.55485487, 4.22119665], [5.650640e+00, 4.86770153, 5.00692511, -3.77362943, 2.78308082, 0.220573112], [-3.25627398, -4.64873171, -2.7951088, -0.643593251, 1.89478385, -3.40890503]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    %3 = "stablehlo.constant"() <{value = dense<[[1, 5, -1, 0, -5, 1], [3, -2, -3, -4, 0, 1], [0, -5, 0, 0, 0, -3], [-2, 1, 0, 0, -4, 2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%2, %3) : (tensor<4x6xf32>, tensor<4x6xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<6xf32>, tensor<6xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[5.650640e+00, 4.86770153, 5.00692511, 3.000000e+00, 3.000000e+00, 4.22119665]> : tensor<6xf32>}> : () -> tensor<6xf32>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -5, -3, -4, -5, -3]> : tensor<6xi32>}> : () -> tensor<6xi32>
    "func.return"(%0, %1) : (tensor<6xf32>, tensor<6xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

