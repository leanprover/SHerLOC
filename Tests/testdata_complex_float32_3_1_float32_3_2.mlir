"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x1xf32>, tensor<3x2xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f32>>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %6 = "stablehlo.complex"(%5, %3#1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xcomplex<f32>>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    "func.return"(%6) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x1xf32>, tensor<3x2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4.849480e-01], [-0.83316183], [3.96992493]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.729111552, -1.21457767], [0.0993311554, -4.24018431], [-1.15859509, -2.34884763]]> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
    "func.return"(%1, %2) : (tensor<3x1xf32>, tensor<3x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.849480e-01,0.729111552), (4.849480e-01,-1.21457767)], [(-0.83316183,0.0993311554), (-0.83316183,-4.24018431)], [(3.96992493,-1.15859509), (3.96992493,-2.34884763)]]> : tensor<3x2xcomplex<f32>>}> : () -> tensor<3x2xcomplex<f32>>
    "func.return"(%0) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

