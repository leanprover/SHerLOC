"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      "stablehlo.return"(%6) : (tensor<complex<f32>>) -> ()
    }) : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(4.09386396,2.19344378), (-3.96042085,-4.04716253), (-0.0249252561,3.71205926)], [(-4.58772039,-0.995992541), (-0.0576205179,0.514490902), (3.764940e+00,-1.50847459)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-0.49385643,1.19745123), (-4.01804113,-3.53267169), (3.74001479,2.20358467)]> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    "func.return"(%0) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

