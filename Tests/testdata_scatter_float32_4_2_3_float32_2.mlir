"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.85092044, -1.29653883, 5.04398203], [-1.77891552, 1.14304197, -0.242220432]], [[0.770432711, 0.497477651, 0.199629322], [1.07187033, 0.0254457798, 1.49424314]], [[-0.667058706, -6.894630e-01, -0.501317859], [0.405911714, -3.601150e+00, 2.04743695]], [[1.35089195, 0.783829689, 0.029527653], [2.21560669, -3.0994556, 0.691326737]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[0.18563509, -2.30085182]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<4x2x3xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.85092044, -1.29653883, 5.04398203], [-1.77891552, 1.14304197, -0.242220432]], [[0.770432711, 0.497477651, 0.199629322], [1.07187033, 0.0254457798, 1.49424314]], [[-0.667058706, -6.894630e-01, -0.501317859], [0.405911714, -3.601150e+00, 2.04743695]], [[1.35089195, 0.783829689, 0.18563509], [2.21560669, -3.0994556, -2.30085182]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    "func.return"(%0) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

