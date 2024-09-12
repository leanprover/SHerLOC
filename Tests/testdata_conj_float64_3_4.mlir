"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f64>>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<3x4xf64>
    %6 = "stablehlo.complex"(%2, %5) : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x4xcomplex<f64>>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    "func.return"(%6) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.8963581247357544, -3.9022169915354232, 2.6657789104597041, 1.7520647517365588], [-5.4495198723021723, -0.30357298841749769, 4.478750797787062, 1.0383215526922465], [-0.040330506564937703, -1.2216473191328965, -2.1613784962585871, -1.0604084893165211]]> : tensor<3x4xf64>}> : () -> tensor<3x4xf64>
    "func.return"(%1) : (tensor<3x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-1.8963581247357544,0.000000e+00), (-3.9022169915354232,0.000000e+00), (2.6657789104597041,0.000000e+00), (1.7520647517365588,0.000000e+00)], [(-5.4495198723021723,0.000000e+00), (-0.30357298841749769,0.000000e+00), (4.478750797787062,0.000000e+00), (1.0383215526922465,0.000000e+00)], [(-0.040330506564937703,0.000000e+00), (-1.2216473191328965,0.000000e+00), (-2.1613784962585871,0.000000e+00), (-1.0604084893165211,0.000000e+00)]]> : tensor<3x4xcomplex<f64>>}> : () -> tensor<3x4xcomplex<f64>>
    "func.return"(%0) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

