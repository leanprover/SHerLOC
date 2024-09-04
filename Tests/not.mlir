"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si4"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[7, -8, 0]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %25 = "stablehlo.not"(%24) : (tensor<3xi4>) -> tensor<3xi4>
    "check.expect_eq_const"(%25) <{value = dense<[-8, 7, -1]> : tensor<3xi4>}> : (tensor<3xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui4"}> ({
    %22 = "stablehlo.constant"() <{value = dense<[0, 7, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    %23 = "stablehlo.not"(%22) : (tensor<3xui4>) -> tensor<3xui4>
    "check.expect_eq_const"(%23) <{value = dense<[15, 8, 0]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si8"}> ({
    %20 = "stablehlo.constant"() <{value = dense<[127, -128, 0]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %21 = "stablehlo.not"(%20) : (tensor<3xi8>) -> tensor<3xi8>
    "check.expect_eq_const"(%21) <{value = dense<[-128, 127, -1]> : tensor<3xi8>}> : (tensor<3xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui8"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[0, 127, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %19 = "stablehlo.not"(%18) : (tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%19) <{value = dense<[255, 128, 0]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si16"}> ({
    %16 = "stablehlo.constant"() <{value = dense<[32767, -32768, 0]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %17 = "stablehlo.not"(%16) : (tensor<3xi16>) -> tensor<3xi16>
    "check.expect_eq_const"(%17) <{value = dense<[-32768, 32767, -1]> : tensor<3xi16>}> : (tensor<3xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui16"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[0, 32767, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    %15 = "stablehlo.not"(%14) : (tensor<3xui16>) -> tensor<3xui16>
    "check.expect_eq_const"(%15) <{value = dense<[65535, 32768, 0]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si32"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[2147483647, -2147483648, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %13 = "stablehlo.not"(%12) : (tensor<3xi32>) -> tensor<3xi32>
    "check.expect_eq_const"(%13) <{value = dense<[-2147483648, 2147483647, -1]> : tensor<3xi32>}> : (tensor<3xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui32"}> ({
    %10 = "stablehlo.constant"() <{value = dense<[0, 2147483647, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %11 = "stablehlo.not"(%10) : (tensor<3xui32>) -> tensor<3xui32>
    "check.expect_eq_const"(%11) <{value = dense<[4294967295, 2147483648, 0]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_si64"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[9223372036854775807, -9223372036854775808, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %9 = "stablehlo.not"(%8) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%9) <{value = dense<[-9223372036854775808, 9223372036854775807, -1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_ui64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    %7 = "stablehlo.not"(%6) : (tensor<3xui64>) -> tensor<3xui64>
    "check.expect_eq_const"(%7) <{value = dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %5 = "stablehlo.not"(%4) : (tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%5) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1_splat_false"}> ({
    %2 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %3 = "stablehlo.not"(%2) : (tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%3) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "not_op_test_i1_splat_true"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.not"(%0) : (tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%1) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

