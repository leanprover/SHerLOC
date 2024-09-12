"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x3xf32>
    %5 = "stablehlo.broadcast_in_dim"(%3#1) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %6 = "stablehlo.remainder"(%3#0, %5) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    "func.return"(%6) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-5.56109047, -1.28362596, 0.951184451], [4.54094934, -1.53028965, 3.18574858], [-1.86437631, -3.51771069, 1.39734781], [-5.04448509, 0.316858649, 0.393427491]], [[4.79267359, -2.66264415, 4.30187178], [-3.77170944, 3.71432328, -1.28224826], [2.37391233, -3.8620162, -3.19630313], [-4.33320427, 1.32255864, 1.83272684]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[0.777158319, -2.80968857, -0.944601118]], [[-0.386181593, 5.995370e-01, 2.75592542]]]> : tensor<2x1x3xf32>}> : () -> tensor<2x1x3xf32>
    "func.return"(%1, %2) : (tensor<2x4x3xf32>, tensor<2x1x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-0.12098223, -1.28362596, 0.00658333301], [0.655157744, -1.53028965, 0.351945221], [-0.310059667, -0.708022118, 0.452746689], [-0.381535172, 0.316858649, 0.393427491]], [[0.158494473, -0.264496088, 1.54594636], [-0.296075106, 0.117101192, -1.28224826], [0.0568227768, -0.264794111, -0.440377712], [-0.085206747, 0.123484612, 1.83272684]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    "func.return"(%0) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

