"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_i1"}> ({
    %48 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %49 = "stablehlo.convert"(%48) : (tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%49) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_si64"}> ({
    %46 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %47 = "stablehlo.convert"(%46) : (tensor<2xi1>) -> tensor<2xi64>
    "check.expect_eq_const"(%47) <{value = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_ui64"}> ({
    %44 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %45 = "stablehlo.convert"(%44) : (tensor<2xi1>) -> tensor<2xui64>
    "check.expect_eq_const"(%45) <{value = dense<[1, 0]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_f64"}> ({
    %42 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %43 = "stablehlo.convert"(%42) : (tensor<2xi1>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%43) <{value = dense<[1.000000e+00, 0.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_i1_to_c128"}> ({
    %40 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %41 = "stablehlo.convert"(%40) : (tensor<2xi1>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%41) <{value = dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_i1"}> ({
    %38 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %39 = "stablehlo.convert"(%38) : (tensor<3xi64>) -> tensor<3xi1>
    "check.expect_eq_const"(%39) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_si64"}> ({
    %36 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %37 = "stablehlo.convert"(%36) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%37) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_ui64"}> ({
    %34 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi4>}> : () -> tensor<2xi4>
    %35 = "stablehlo.convert"(%34) : (tensor<2xi4>) -> tensor<2xui64>
    "check.expect_eq_const"(%35) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_f64"}> ({
    %32 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %33 = "stablehlo.convert"(%32) : (tensor<3xi64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%33) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_si64_to_c128"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[-1, 0, 1]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %31 = "stablehlo.convert"(%30) : (tensor<3xi4>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%31) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_i1"}> ({
    %28 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %29 = "stablehlo.convert"(%28) : (tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%29) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_si64"}> ({
    %26 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %27 = "stablehlo.convert"(%26) : (tensor<2xui64>) -> tensor<2xi64>
    "check.expect_eq_const"(%27) <{value = dense<[0, 1]> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_ui64"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %25 = "stablehlo.convert"(%24) : (tensor<2xui4>) -> tensor<2xui64>
    "check.expect_eq_const"(%25) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_f64"}> ({
    %22 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %23 = "stablehlo.convert"(%22) : (tensor<2xui64>) -> tensor<2xf64>
    "check.expect_almost_eq_const"(%23) <{value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>}> : (tensor<2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_ui64_to_c128"}> ({
    %20 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %21 = "stablehlo.convert"(%20) : (tensor<2xui4>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%21) <{value = dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_i1"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %19 = "stablehlo.convert"(%18) : (tensor<3xf64>) -> tensor<3xi1>
    "check.expect_eq_const"(%19) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_si64"}> ({
    %16 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %17 = "stablehlo.convert"(%16) : (tensor<3xf64>) -> tensor<3xi64>
    "check.expect_eq_const"(%17) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_ui64"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %15 = "stablehlo.convert"(%14) : (tensor<2xf64>) -> tensor<2xui64>
    "check.expect_eq_const"(%15) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_f64"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %13 = "stablehlo.convert"(%12) : (tensor<3xf64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%13) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_f64_to_c128"}> ({
    %10 = "stablehlo.constant"() <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %11 = "stablehlo.convert"(%10) : (tensor<3xf64>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%11) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_i1"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %9 = "stablehlo.convert"(%8) : (tensor<3xcomplex<f64>>) -> tensor<3xi1>
    "check.expect_eq_const"(%9) <{value = dense<[true, false, true]> : tensor<3xi1>}> : (tensor<3xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_si64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %7 = "stablehlo.convert"(%6) : (tensor<3xcomplex<f64>>) -> tensor<3xi64>
    "check.expect_eq_const"(%7) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_ui64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[(0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %5 = "stablehlo.convert"(%4) : (tensor<2xcomplex<f64>>) -> tensor<2xui64>
    "check.expect_eq_const"(%5) <{value = dense<[0, 1]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %3 = "stablehlo.convert"(%2) : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "convert_op_test_c128_to_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    %1 = "stablehlo.convert"(%0) : (tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-1.000000e+00,0.000000e+00), (0.000000e+00,1.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<3xcomplex<f64>>}> : (tensor<3xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

