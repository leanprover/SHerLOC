"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_si4"}> ({
    %36 = "stablehlo.constant"() <{value = dense<[7, -8, -8]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %37 = "stablehlo.constant"() <{value = dense<[0, 7, -8]> : tensor<3xi4>}> : () -> tensor<3xi4>
    %38 = "stablehlo.xor"(%36, %37) : (tensor<3xi4>, tensor<3xi4>) -> tensor<3xi4>
    "check.expect_eq_const"(%38) <{value = dense<[7, -1, 0]> : tensor<3xi4>}> : (tensor<3xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_ui4"}> ({
    %33 = "stablehlo.constant"() <{value = dense<[0, 7, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    %34 = "stablehlo.constant"() <{value = dense<15> : tensor<3xui4>}> : () -> tensor<3xui4>
    %35 = "stablehlo.xor"(%33, %34) : (tensor<3xui4>, tensor<3xui4>) -> tensor<3xui4>
    "check.expect_eq_const"(%35) <{value = dense<[15, 8, 0]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_si8"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[127, -128, -128]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %31 = "stablehlo.constant"() <{value = dense<[0, 127, -128]> : tensor<3xi8>}> : () -> tensor<3xi8>
    %32 = "stablehlo.xor"(%30, %31) : (tensor<3xi8>, tensor<3xi8>) -> tensor<3xi8>
    "check.expect_eq_const"(%32) <{value = dense<[127, -1, 0]> : tensor<3xi8>}> : (tensor<3xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_ui8"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[0, 127, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %28 = "stablehlo.constant"() <{value = dense<255> : tensor<3xui8>}> : () -> tensor<3xui8>
    %29 = "stablehlo.xor"(%27, %28) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%29) <{value = dense<[255, 128, 0]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_si16"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[32767, -32768, -32768]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %25 = "stablehlo.constant"() <{value = dense<[0, 32767, -32768]> : tensor<3xi16>}> : () -> tensor<3xi16>
    %26 = "stablehlo.xor"(%24, %25) : (tensor<3xi16>, tensor<3xi16>) -> tensor<3xi16>
    "check.expect_eq_const"(%26) <{value = dense<[32767, -1, 0]> : tensor<3xi16>}> : (tensor<3xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_ui16"}> ({
    %21 = "stablehlo.constant"() <{value = dense<[0, 32767, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    %22 = "stablehlo.constant"() <{value = dense<65535> : tensor<3xui16>}> : () -> tensor<3xui16>
    %23 = "stablehlo.xor"(%21, %22) : (tensor<3xui16>, tensor<3xui16>) -> tensor<3xui16>
    "check.expect_eq_const"(%23) <{value = dense<[65535, 32768, 0]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_si32"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %19 = "stablehlo.constant"() <{value = dense<[0, 2147483647, -2147483648]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %20 = "stablehlo.xor"(%18, %19) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
    "check.expect_eq_const"(%20) <{value = dense<[2147483647, -1, 0]> : tensor<3xi32>}> : (tensor<3xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_ui32"}> ({
    %15 = "stablehlo.constant"() <{value = dense<[0, 2147483647, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %16 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<3xui32>}> : () -> tensor<3xui32>
    %17 = "stablehlo.xor"(%15, %16) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    "check.expect_eq_const"(%17) <{value = dense<[4294967295, 2147483648, 0]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_si64"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %13 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %14 = "stablehlo.xor"(%12, %13) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%14) <{value = dense<[9223372036854775807, -1, 0]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_ui64"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    %10 = "stablehlo.constant"() <{value = dense<18446744073709551615> : tensor<3xui64>}> : () -> tensor<3xui64>
    %11 = "stablehlo.xor"(%9, %10) : (tensor<3xui64>, tensor<3xui64>) -> tensor<3xui64>
    "check.expect_eq_const"(%11) <{value = dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_i1"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %7 = "stablehlo.constant"() <{value = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %8 = "stablehlo.xor"(%6, %7) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%8) <{value = dense<[false, true, true, false]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_i1_splat_false"}> ({
    %3 = "stablehlo.constant"() <{value = dense<false> : tensor<2xi1>}> : () -> tensor<2xi1>
    %4 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %5 = "stablehlo.xor"(%3, %4) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%5) <{value = dense<[false, true]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "xor_op_test_i1_splat_true"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<2xi1>}> : () -> tensor<2xi1>
    %1 = "stablehlo.constant"() <{value = dense<[false, true]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %2 = "stablehlo.xor"(%0, %1) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

