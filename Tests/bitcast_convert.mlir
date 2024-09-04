"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "bitcast_convert_op_test_i1_to_i64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[true, true, true, true, false, true, true, true, true, false, true, true, false, false, true, true, true, true, false, true, false, true, false, true, true, false, false, true, false, false, false, true, true, true, true, false, false, true, true, false, true, false, true, false, false, false, true, false, true, true, false, false, false, true, false, false, true, false, false, false, false, false, false, false]> : tensor<64xi1>}> : () -> tensor<64xi1>
    %7 = "stablehlo.bitcast_convert"(%6) : (tensor<64xi1>) -> tensor<i64>
    "check.expect_eq_const"(%7) <{value = dense<81985529216486895> : tensor<i64>}> : (tensor<i64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "bitcast_convert_op_test_i64_to_f64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<81985529216486895> : tensor<i64>}> : () -> tensor<i64>
    %5 = "stablehlo.bitcast_convert"(%4) : (tensor<i64>) -> tensor<f64>
    "check.expect_almost_eq_const"(%5) <{value = dense<3.5127005640885037E-303> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "bitcast_convert_op_test_f64_to_i1"}> ({
    %2 = "stablehlo.constant"() <{value = dense<3.5127005640885037E-303> : tensor<f64>}> : () -> tensor<f64>
    %3 = "stablehlo.bitcast_convert"(%2) : (tensor<f64>) -> tensor<64xi1>
    "check.expect_eq_const"(%3) <{value = dense<[true, true, true, true, false, true, true, true, true, false, true, true, false, false, true, true, true, true, false, true, false, true, false, true, true, false, false, true, false, false, false, true, true, true, true, false, false, true, true, false, true, false, true, false, false, false, true, false, true, true, false, false, false, true, false, false, true, false, false, false, false, false, false, false]> : tensor<64xi1>}> : (tensor<64xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "bitcast_convert_op_test_c128_to_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(3.5127005640885037E-303,1.4146638603141378E-315)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.bitcast_convert"(%0) : (tensor<complex<f64>>) -> tensor<2xcomplex<f32>>
    "check.expect_eq_const"(%1) <{value = dense<[(-4.13604116E-33,2.99881655E-38), (1.14437421E-28,0.000000e+00)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

