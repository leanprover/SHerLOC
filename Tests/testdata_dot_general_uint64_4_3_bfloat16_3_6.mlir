"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui64>, tensor<3x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui64>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xbf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui64>, tensor<3x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 1, 3], [5, 0, 5], [3, 1, 2], [1, 3, 2]]> : tensor<4x3xui64>}> : () -> tensor<4x3xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[1.429690e+00, -1.265630e+00, -1.318360e-01, -1.101560e+00, 2.796880e+00, 2.421880e+00], [-3.339840e-01, 2.734380e+00, 4.062500e+00, -1.867190e+00, -1.117190e+00, -1.281250e+00], [2.250000e+00, -3.125000e+00, 1.476560e+00, -8.007810e-02, 3.765630e+00, 9.609370e-01]]> : tensor<3x6xbf16>}> : () -> tensor<3x6xbf16>
    "func.return"(%1, %2) : (tensor<4x3xui64>, tensor<3x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.356250e+01, -1.300000e+01, 7.843750e+00, -7.625000e+00, 2.412500e+01, 1.368750e+01], [1.837500e+01, -2.200000e+01, 6.718750e+00, -5.906250e+00, 3.275000e+01, 1.687500e+01], [8.437500e+00, -7.312500e+00, 6.625000e+00, -5.343750e+00, 1.481250e+01, 7.906250e+00], [4.937500e+00, 6.875000e-01, 1.500000e+01, -6.875000e+00, 6.968750e+00, 5.000000e-01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

