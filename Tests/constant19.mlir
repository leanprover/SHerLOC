"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

