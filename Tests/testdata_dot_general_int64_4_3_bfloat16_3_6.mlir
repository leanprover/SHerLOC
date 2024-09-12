"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi64>, tensor<3x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi64>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xbf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi64>, tensor<3x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 9, 2], [0, -4, 0], [-1, -2, 0], [-3, 2, -3]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.812500e+00, -4.785160e-01, -6.531250e+00, -3.531250e+00, -2.500000e+00, 7.734380e-01], [-2.546880e+00, 4.343750e+00, -2.140630e+00, -8.515620e-01, 2.484380e+00, 2.890630e+00], [1.882810e+00, -1.328130e+00, 1.750000e+00, -5.625000e-01, -1.078130e+00, 6.031250e+00]]> : tensor<3x6xbf16>}> : () -> tensor<3x6xbf16>
    "func.return"(%1, %2) : (tensor<4x3xi64>, tensor<3x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.475000e+01, 3.550000e+01, -2.887500e+01, -1.587500e+01, 1.518750e+01, 3.950000e+01], [1.018750e+01, -1.737500e+01, 8.562500e+00, 3.406250e+00, -9.937500e+00, -1.156250e+01], [7.906250e+00, -8.187500e+00, 1.081250e+01, 5.250000e+00, -2.468750e+00, -6.562500e+00], [-2.312500e+00, 1.412500e+01, 1.006250e+01, 1.056250e+01, 1.568750e+01, -1.462500e+01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

