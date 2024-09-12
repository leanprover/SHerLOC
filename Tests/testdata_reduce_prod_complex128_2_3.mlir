"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f64>>
    %4 = "stablehlo.constant"() <{value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>):
      %6 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
      "stablehlo.return"(%6) : (tensor<complex<f64>>) -> ()
    }) : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-3.7075252346119498,2.3747053267110676), (-2.8248724820127937,-0.43379307192698208), (-4.3787338274375687,3.667746913686682)], [(1.8555001818174246,1.886921310804297), (0.52301083883203736,-2.0290334106508694), (-1.5719747328314484,0.35950850655637528)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-11.360195834766756,-2.5895422100587964), (-2.3576195626597731,5.5048821684041487), (5.5646727231599433,-7.3397975336459531)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    "func.return"(%0) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

