"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_c128_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(0x7FF0000000000001,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<(0x7FF0000000000001,-0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

