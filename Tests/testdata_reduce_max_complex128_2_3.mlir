"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f64>>
    %4 = "stablehlo.constant"() <{value = dense<(0xFFF0000000000000,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>):
      %6 = "stablehlo.real"(%arg0) : (tensor<complex<f64>>) -> tensor<f64>
      %7 = "stablehlo.real"(%arg1) : (tensor<complex<f64>>) -> tensor<f64>
      %8 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %9 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %10 = "stablehlo.imag"(%arg0) : (tensor<complex<f64>>) -> tensor<f64>
      %11 = "stablehlo.imag"(%arg1) : (tensor<complex<f64>>) -> tensor<f64>
      %12 = "stablehlo.compare"(%10, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %13 = "stablehlo.select"(%8, %12, %9) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
      %14 = "stablehlo.select"(%13, %arg0, %arg1) : (tensor<i1>, tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
      "stablehlo.return"(%14) : (tensor<complex<f64>>) -> ()
    }) : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(2.7669833167087874,-0.94169600484289995), (-2.3719987378004435,3.3066471553557681), (-5.8535964049536684,-7.1686460856967784)], [(0.1492160216132154,1.7900065296260239), (-5.2811226168470782,2.9759432261118173), (1.0735368743006275,-0.012922089414236853)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(2.7669833167087874,-0.94169600484289995), (-2.3719987378004435,3.3066471553557681), (1.0735368743006275,-0.012922089414236853)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    "func.return"(%0) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

