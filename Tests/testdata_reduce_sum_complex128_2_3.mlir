"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f64>>
    %4 = "stablehlo.constant"() <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>):
      %6 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
      "stablehlo.return"(%6) : (tensor<complex<f64>>) -> ()
    }) : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-5.0016637729619955,-2.7226144432763526), (0.30071536304071916,-0.31702764635348196), (2.2561604067910577,-0.52579407344003848)], [(-0.38841319310607747,4.0382795230843636), (0.52697596861662555,2.4393389058990746), (0.31194372240006796,0.59519467929679115)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-5.3900769660680732,1.315665079808011), (0.82769133165734465,2.1223112595455929), (2.5681041291911257,0.069400605856752673)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    "func.return"(%0) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

