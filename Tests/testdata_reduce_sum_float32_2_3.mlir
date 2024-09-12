"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%5) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.0074451, 3.005660e+00, 1.72375727], [-0.16611506, -4.26857853, 2.59610081]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.841330051, -1.26291847, 4.31985807]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

