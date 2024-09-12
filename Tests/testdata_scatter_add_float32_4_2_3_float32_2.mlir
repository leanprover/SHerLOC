"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.823510e+00, 0.423392832, 2.67835927], [-2.08515167, -3.6946454, 2.47799826]], [[-2.64480543, -0.643004477, 5.69479227], [-3.7926693, -1.77346087, -1.33025837]], [[-1.96124375, -0.377599716, -3.10537505], [-3.95107126, -1.77462971, 2.59804296]], [[-6.70349025, -0.822746276, 2.51327872], [-1.05100095, 3.29730201, 5.58889532]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[5.25133562, -1.03823602]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<4x2x3xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.823510e+00, 0.423392832, 2.67835927], [-2.08515167, -3.6946454, 2.47799826]], [[-2.64480543, -0.643004477, 5.69479227], [-3.7926693, -1.77346087, -1.33025837]], [[-1.96124375, -0.377599716, -3.10537505], [-3.95107126, -1.77462971, 2.59804296]], [[-6.70349025, -0.822746276, 7.76461411], [-1.05100095, 3.29730201, 4.55065918]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    "func.return"(%0) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

