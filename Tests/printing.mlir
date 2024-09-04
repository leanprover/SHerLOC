"builtin.module"() ({
  "func.func"() <{function_type = () -> (tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, -3, -4]> : tensor<4xi8>}> : () -> tensor<4xi8>
    %1 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]> : tensor<4xf16>}> : () -> tensor<4xf16>
    %2 = "stablehlo.constant"() <{value = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<ui8>}> : () -> tensor<ui8>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>}> : () -> tensor<2x2xi8>
    %6 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %7 = "stablehlo.constant"() <{value = dense<> : tensor<0xui8>}> : () -> tensor<0xui8>
    %8 = "stablehlo.constant"() <{value = dense<> : tensor<0x2xi8>}> : () -> tensor<0x2xi8>
    %9 = "stablehlo.constant"() <{value = dense<> : tensor<2x0xi8>}> : () -> tensor<2x0xi8>
    "func.return"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9) : (tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>) -> ()
  }) : () -> ()
}) : () -> ()

